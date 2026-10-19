// v21 algorithm re-expressed for the measured MUSA mp_31 WMMA layouts.
// Controlled tuning candidate: compile-time ablations, same BF16/FP32 contract.
// Training candidate: pre-scaled Q and base-2 log-sum-exp for Triton backward.
#ifndef V21_PRESCALED
#error This training ABI requires pre-scaled Q
#endif
#include <musa_runtime.h>
#include <musa_bf16.h>
#include <mma.h>
#include <cfloat>
using namespace mtmusa;
using bf16=__mt_bfloat16;
#ifdef V21_BIAS2_BF16
using bias2_t=bf16;
#else
using bias2_t=float;
#endif
constexpr float LOG2E=1.4426950408889634f;
#ifdef V21_PRESCALED
constexpr float QSCALE=1.f;
#else
constexpr float QSCALE=.17677669529663689f;
#endif
#ifndef V21_BLOCK_M
#define V21_BLOCK_M 128
#endif
__device__ float max8(float x) {
    x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,1));
    x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,2));
    return fmaxf(x,__shfl_xor_sync(0xffffffff,x,4));
}
__device__ float sum8(float x) {
    x+=__shfl_xor_sync(0xffffffff,x,1);
    x+=__shfl_xor_sync(0xffffffff,x,2);
    return x+__shfl_xor_sync(0xffffffff,x,4);
}
#ifndef V21_WARPS
#define V21_WARPS 4
#endif
#ifndef V21_KSTRIDE
#define V21_KSTRIDE 32
#endif
#ifndef V21_QSTRIDE
#define V21_QSTRIDE 40
#endif
#ifdef V21_FAST_EXP2
#define V21_EXP(x) __exp2f(x)
#else
#define V21_EXP(x) exp2f(x)
#endif
#ifndef V21_PSTRIDE
#define V21_PSTRIDE 32
#endif
__device__ __forceinline__ unsigned packbf(float a,float b) {
    return unsigned(__bfloat16_as_ushort(__float2bfloat16(a))) |
           (unsigned(__bfloat16_as_ushort(__float2bfloat16(b)))<<16);
}
template<int BM,int FS=0>
__global__ void tri_mapped(const bf16* Q,const bf16* K,const bf16* V,
    const float* B1,const bias2_t* B2,bf16* O,float* M,int runtimeS,int N) {
    const int S=FS?FS:runtimeS;
    constexpr int NT=V21_WARPS*32, QT=BM/(V21_WARPS*16), BN=32, KS=V21_KSTRIDE;
    static_assert(BM%(V21_WARPS*16)==0,"tile/warp mismatch");
    __shared__ bf16 sq[BM*V21_QSTRIDE];
    __shared__ bf16 sk[BN*KS],sv[BN*KS];
    __shared__ float sb[BN];
    __shared__ float sout[BM*32];
#ifdef V21_SHARED_P
    __shared__ bf16 sp[BM*V21_PSTRIDE];
#endif
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const int n=blockIdx.z,b=blockIdx.y/4,h=blockIdx.y%4,m=blockIdx.x*BM;
    const int base=((b*N+n)*S*4+h)*32;
    Q+=base;K+=base;V+=base;O+=base;
    B1+=(b*N+n)*S;B2+=(b*4+h)*S*S;
    for(int i=tid;i<BM*4;i+=NT) {
        int row=i/4,col=(i%4)*8;
        int4 x=make_int4(0,0,0,0);
        if((FS && FS%BM==0)||m+row<S)x=*reinterpret_cast<const int4*>(Q+(m+row)*128+col);
        *reinterpret_cast<int4*>(sq+row*V21_QSTRIDE+col)=x;
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a,16,16,32,bf16,wmma::row_major> q[QT];
    wmma::fragment<wmma::accumulator,16,16,32,float> out[QT][2];
    float maxv[QT][4],sumv[QT][4];
    #pragma unroll
    for(int t=0;t<QT;t++){
        for(int r=0;r<4;r++){maxv[t][r]=-FLT_MAX;sumv[t][r]=0.f;}
        #pragma unroll
        for(int i=0;i<8;i++) {
            int row=lane/4+8*(i/4),col=2*(lane%4)+8*(i%4);
            const bf16* src=sq+((warp*QT+t)*16+row)*V21_QSTRIDE+col;
            reinterpret_cast<unsigned*>(&q[t])[i]=packbf(__bfloat162float(src[0])*QSCALE,__bfloat162float(src[1])*QSCALE);
        }
        for(int d=0;d<2;d++)wmma::fill_fragment(out[t][d],0.f);
    }
#ifdef V21_UNROLL_KB
    #pragma unroll
#elif defined(V21_UNROLL_KB4)
    #pragma unroll 4
#endif
    for(int kb=0;kb<S;kb+=BN){
        for(int i=tid;i<BN*4;i+=NT){
            int r=i/4,c=(i%4)*8;
            int4 x=make_int4(0,0,0,0),y=x;
            if((FS && FS%32==0)||kb+r<S){x=*reinterpret_cast<const int4*>(K+(kb+r)*128+c);y=*reinterpret_cast<const int4*>(V+(kb+r)*128+c);}
            *reinterpret_cast<int4*>(sk+r*KS+c)=x;
            *reinterpret_cast<int4*>(sv+r*KS+c)=y;
        }
        if(tid<BN) {
            if((FS && FS%32==0)||kb+tid<S) {
#ifdef V21_NO_BIAS1
                sb[tid]=0.f;
#else
                sb[tid]=B1[kb+tid];
#endif
            } else sb[tid]=-1e6f;
        }
        __syncthreads();
        #pragma unroll
        for(int t=0;t<QT;t++){
            int wr=(warp*QT+t)*16;
            wmma::fragment<wmma::accumulator,16,16,32,float> score[2];
            #pragma unroll
            for(int c=0;c<2;c++){
                #pragma unroll
                for(int e=0;e<8;e++){
                    int row=m+wr+lane/8+(e/2)*4;
                    int col=kb+c*16+lane%8+(e%2)*8;
                    if((FS && FS%BM==0 && FS%32==0)||(row<S&&col<S)) {
#ifdef V21_BIAS2_BF16
                        score[c].x[e]=sb[col-kb]+__bfloat162float(B2[row*S+col]);
#else
                        score[c].x[e]=sb[col-kb]+B2[row*S+col];
#endif
                    } else score[c].x[e]=-1e6f;
                }
                wmma::fragment<wmma::matrix_b,16,16,32,bf16,wmma::col_major> kf;
                #pragma unroll
                for(int i=0;i<8;i++) {
                    int row=lane/4+8*(i/4),col=2*(lane%4)+8*(i%4);
                    reinterpret_cast<unsigned*>(&kf)[i]=*reinterpret_cast<const unsigned*>(sk+(c*16+row)*KS+col);
                }
                wmma::mma_sync(score[c],q[t],kf,score[c]);
            }
            #pragma unroll
            for(int r=0;r<4;r++){
                float mx=-FLT_MAX;
                #pragma unroll
                for(int c=0;c<2;c++){
                    score[c].x[2*r]*=LOG2E;score[c].x[2*r+1]*=LOG2E;
                    mx=fmaxf(mx,fmaxf(score[c].x[2*r],score[c].x[2*r+1]));
                }
                mx=fmaxf(maxv[t][r],max8(mx));
                float alpha=V21_EXP(maxv[t][r]-mx),sm=0.f;
                #pragma unroll
                for(int c=0;c<2;c++){
                    score[c].x[2*r]=V21_EXP(score[c].x[2*r]-mx);
                    score[c].x[2*r+1]=V21_EXP(score[c].x[2*r+1]-mx);
                    sm+=score[c].x[2*r]+score[c].x[2*r+1];
                    out[t][c].x[2*r]*=alpha;out[t][c].x[2*r+1]*=alpha;
                }
                sumv[t][r]=sumv[t][r]*alpha+sum8(sm);maxv[t][r]=mx;
            }
            wmma::fragment<wmma::matrix_a,16,16,32,bf16,wmma::row_major> p;
#ifdef V21_SHARED_P
            #pragma unroll
            for(int c=0;c<2;c++)for(int e=0;e<8;e++) {
                int row=wr+lane/8+4*(e/2),col=c*16+lane%8+8*(e%2);
                sp[row*V21_PSTRIDE+col]=__float2bfloat16(score[c].x[e]);
            }
#ifdef V21_WARP_P_SYNC
            __syncwarp();
#else
            __syncthreads();
#endif
            #pragma unroll
            for(int i=0;i<8;i++){
                int row=lane/4+8*(i/4),col=2*(lane%4)+8*(i%4);
                reinterpret_cast<unsigned*>(&p)[i]=*reinterpret_cast<const unsigned*>(sp+(wr+row)*V21_PSTRIDE+col);
            }
#else
            #pragma unroll
            for(int e=0;e<16;e++){
                int row=lane/4+8*(e/8),col=2*(lane%4)+(e%2)+8*((e%8)/2);
                int src=(row%4)*8+col%8;
                float lo=__shfl_sync(0xffffffff,score[(e%8)/4].x[(e/8)*4+((e%4)/2)],src);
                float hi=__shfl_sync(0xffffffff,score[(e%8)/4].x[(e/8)*4+2+((e%4)/2)],src);
                p.x[e]=__float2bfloat16(lane<16?lo:hi);
            }
#endif
            #pragma unroll
            for(int d=0;d<2;d++){
                wmma::fragment<wmma::matrix_b,16,16,32,bf16,wmma::row_major> vf;
                wmma::load_matrix_sync(vf,sv+d*16,KS);
                wmma::mma_sync(out[t][d],p,vf,out[t][d]);
            }
        }
        __syncthreads();
    }
#ifndef V21_NO_LSE
    if((lane & 7)==0) {
        #pragma unroll
        for(int t=0;t<QT;t++)for(int r=0;r<4;r++) {
            int row=m+(warp*QT+t)*16+lane/8+4*r;
            if(row<S) M[((b*N+n)*S+row)*4+h]=maxv[t][r]+
#ifdef V21_FAST_LOG2
                __log2f(sumv[t][r]);
#else
                log2f(sumv[t][r]);
#endif
        }
    }
#endif
#ifdef V21_DIRECT_OUT
    #pragma unroll
    for(int t=0;t<QT;t++)for(int d=0;d<2;d++)for(int e=0;e<8;e++){
        int row=m+(warp*QT+t)*16+lane/8+4*(e/2),col=d*16+lane%8+8*(e%2);
        if(row<S)O[row*128+col]=__float2bfloat16(out[t][d].x[e]/sumv[t][e/2]);
    }
#else
    #pragma unroll
    for(int t=0;t<QT;t++){
        #pragma unroll
        for(int d=0;d<2;d++){
            #pragma unroll
            for(int e=0;e<8;e++)out[t][d].x[e]/=sumv[t][e/2];
            wmma::store_matrix_sync(sout+(warp*QT+t)*16*32+d*16,out[t][d],32,wmma::mem_row_major);
        }
    }
    __syncthreads();
    for(int i=tid;i<BM*4;i+=NT){
        int r=i/4,c=(i%4)*8;
        if((FS && FS%BM==0)||m+r<S){
#ifdef V21_DIRECT_FRAG
            #pragma unroll
            for(int e=0;e<4;e++)
                reinterpret_cast<unsigned*>(O+(m+r)*128+c)[e]=packbf(sout[r*32+c+2*e],sout[r*32+c+2*e+1]);
#else
            int4 pack;bf16* vals=reinterpret_cast<bf16*>(&pack);
            #pragma unroll
            for(int e=0;e<8;e++) vals[e]=__float2bfloat16(sout[r*32+c+e]);
            *reinterpret_cast<int4*>(O+(m+r)*128+c)=pack;
#endif
        }
    }
#endif
}
extern "C" void launch_triangle_attention_fwd_musa_lse(const void* q,const void* k,const void* v,
 const void* b1,const void* b2,void* out,void* lse,int BN,int S,int H,int D,int N,void* stream){
    if(H!=4||D!=32||S<=0||N<=0||BN%N)return;
    dim3 grid((S+V21_BLOCK_M-1)/V21_BLOCK_M,(BN/N)*4,N);
#ifdef V21_SPECIAL384
    if(S==384) {
        tri_mapped<V21_BLOCK_M,384><<<grid,V21_WARPS*32,0,(musaStream_t)stream>>>((const bf16*)q,(const bf16*)k,(const bf16*)v,(const float*)b1,(const bias2_t*)b2,(bf16*)out,(float*)lse,S,N);
        return;
    }
#endif
#ifdef V21_SPECIAL512
    if(S==512) {
        tri_mapped<V21_BLOCK_M,512><<<grid,V21_WARPS*32,0,(musaStream_t)stream>>>((const bf16*)q,(const bf16*)k,(const bf16*)v,(const float*)b1,(const bias2_t*)b2,(bf16*)out,(float*)lse,S,N);
        return;
    }
#endif
    tri_mapped<V21_BLOCK_M><<<grid,V21_WARPS*32,0,(musaStream_t)stream>>>((const bf16*)q,(const bf16*)k,(const bf16*)v,(const float*)b1,(const bias2_t*)b2,(bf16*)out,(float*)lse,S,N);
}
