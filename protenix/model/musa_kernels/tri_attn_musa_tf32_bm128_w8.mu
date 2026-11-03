#include <musa_runtime.h>
#include <mma.h>
#include <cfloat>
#include <cmath>
using namespace mtmusa;
constexpr int BM=128, BN=16, WARPS=8, D=32;

__device__ __forceinline__ float max8(float x) {
    x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,1));
    x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,2));
    return fmaxf(x,__shfl_xor_sync(0xffffffff,x,4));
}
__device__ __forceinline__ float sum8(float x) {
    x+=__shfl_xor_sync(0xffffffff,x,1);
    x+=__shfl_xor_sync(0xffffffff,x,2);
    return x+__shfl_xor_sync(0xffffffff,x,4);
}

__global__ void tri_tf32_v2(const float* Q,const float* K,const float* V,
 const float* B1,const float* B2,float* O,float* LSE,int S,int N) {
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
    const int n=blockIdx.z,b=blockIdx.y/4,h=blockIdx.y%4,m=blockIdx.x*BM,wr=warp*16;
    const int base=((b*N+n)*S*4+h)*D;
    Q+=base;K+=base;V+=base;O+=base;B1+=(b*N+n)*S;B2+=(b*4+h)*S*S;
    __shared__ float sq[BM*D],sk[BN*D],sv[BN*D],sp[BM*BN],sout[BM*D];
    for(int i=tid;i<BM*D;i+=WARPS*32){int r=i/D,c=i%D;sq[i]=(m+r<S)?Q[(m+r)*4*D+c]:0.f;}
    __syncthreads();
    wmma::fragment<wmma::matrix_a,16,16,16,wmma::precision::tf32,wmma::row_major> qf[2];
    #pragma unroll
    for(int d=0;d<2;d++)wmma::load_matrix_sync(qf[d],sq+wr*D+d*16,D);
    wmma::fragment<wmma::accumulator,16,16,16,float> out[2];
    wmma::fill_fragment(out[0],0.f);wmma::fill_fragment(out[1],0.f);
    float maxv[4]={-FLT_MAX,-FLT_MAX,-FLT_MAX,-FLT_MAX};
    float sumv[4]={0.f,0.f,0.f,0.f};
    for(int kb=0;kb<S;kb+=BN){
        for(int i=tid;i<BN*D;i+=WARPS*32){
            int r=i/D,c=i%D;
            sk[i]=(kb+r<S)?K[(kb+r)*4*D+c]:0.f;
            sv[i]=(kb+r<S)?V[(kb+r)*4*D+c]:0.f;
        }
        __syncthreads();
        wmma::fragment<wmma::accumulator,16,16,16,float> score;
        wmma::fill_fragment(score,0.f);
        #pragma unroll
        for(int d=0;d<2;d++){
            wmma::fragment<wmma::matrix_b,16,16,16,wmma::precision::tf32,wmma::col_major> kf;
            wmma::load_matrix_sync(kf,sk+d*16,D);wmma::mma_sync(score,qf[d],kf,score);
        }
        #pragma unroll
        for(int e=0;e<8;e++){
            int row=m+wr+lane/8+(e/2)*4,col=kb+lane%8+(e%2)*8;
            score.x[e]=(row<S&&col<S)?score.x[e]+B1[col]+B2[row*S+col]:-1e6f;
        }
        #pragma unroll
        for(int r=0;r<4;r++){
            float mx=max8(fmaxf(score.x[2*r],score.x[2*r+1]));
            mx=fmaxf(mx,maxv[r]);float alpha=__expf(maxv[r]-mx);
            score.x[2*r]=__expf(score.x[2*r]-mx);
            score.x[2*r+1]=__expf(score.x[2*r+1]-mx);
            float sm=sum8(score.x[2*r]+score.x[2*r+1]);
            sumv[r]=sumv[r]*alpha+sm;maxv[r]=mx;
            out[0].x[2*r]*=alpha;out[0].x[2*r+1]*=alpha;
            out[1].x[2*r]*=alpha;out[1].x[2*r+1]*=alpha;
        }
        #pragma unroll
        for(int e=0;e<8;e++){
            int row=wr+lane/8+(e/2)*4,col=lane%8+(e%2)*8;
            sp[row*BN+col]=score.x[e];
        }
        __syncwarp();
        wmma::fragment<wmma::matrix_a,16,16,16,wmma::precision::tf32,wmma::row_major> pf;
        wmma::load_matrix_sync(pf,sp+wr*BN,BN);
        #pragma unroll
        for(int d=0;d<2;d++){
            wmma::fragment<wmma::matrix_b,16,16,16,wmma::precision::tf32,wmma::row_major> vf;
            wmma::load_matrix_sync(vf,sv+d*16,D);wmma::mma_sync(out[d],pf,vf,out[d]);
        }
        __syncthreads();
    }
    #pragma unroll
    for(int d=0;d<2;d++)for(int e=0;e<8;e++)out[d].x[e]/=sumv[e/2];
    wmma::store_matrix_sync(sout+wr*D,out[0],D,wmma::mem_row_major);
    wmma::store_matrix_sync(sout+wr*D+16,out[1],D,wmma::mem_row_major);
    __syncwarp();
    for(int i=lane;i<16*D;i+=32){int r=i/D,c=i%D,gr=m+wr+r;if(gr<S)O[gr*4*D+c]=sout[(wr+r)*D+c];}
    if((lane&7)==0)for(int r=0;r<4;r++){
        int gr=m+wr+lane/8+4*r;
        if(gr<S)LSE[((b*N+n)*S+gr)*4+h]=maxv[r]+logf(sumv[r]);
    }
}
extern "C" void launch_triangle_attention_tf32_v2(const void*q,const void*k,const void*v,
 const void*b1,const void*b2,void*out,void*lse,int BNtotal,int S,int H,int head_dim,int N,void*stream){
    if(H!=4||head_dim!=32||S<=0||N<=0||BNtotal%N)return;
    dim3 grid((S+BM-1)/BM,(BNtotal/N)*4,N);
    tri_tf32_v2<<<grid,WARPS*32,0,(musaStream_t)stream>>>((const float*)q,(const float*)k,
      (const float*)v,(const float*)b1,(const float*)b2,(float*)out,(float*)lse,S,N);
}
