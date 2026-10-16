"""Install the two image-provided extensions, then build the v21 source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARTIFACTS = {
    "protenix/model/layer_norm/fast_layer_norm_cuda_v2.so":
        "d3993c3b0aae9d0e828450af0eea98787a6b3db1524ce6e99523b89ed1359864",
    "protenix/model/musa_kernels/mutlass_triangle_bmm/protenix_mutlass_triangle_bmm_v5.so":
        "46ff8e3410594f9273cab604205137990c59aad8df3f1a3632b0412181a5aeaa",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/app/protenix"),
                        help="Source tree of the tested official MUSA image")
    args = parser.parse_args()
    selected = []
    for relative, expected in ARTIFACTS.items():
        target = REPO / relative
        source = target if target.is_file() else args.source_root / relative
        if not source.is_file():
            parser.error(f"Missing official image artifact: {source}")
        if digest(source) != expected:
            parser.error(f"SHA256 mismatch for {source}; use the tested image/ABI")
        selected.append((source, target))
    for source, target in selected:
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    subprocess.run(["bash", str(REPO / "protenix/model/tri_attention/v21/build_v21_forward_fast_p36q32_repro.sh")], check=True)
    native = REPO / "protenix/model/tri_attention/v21/libv21_training_fast_p36q32_repro.so"
    metadata = {str(dst.relative_to(REPO)): digest(dst) for _, dst in selected}
    metadata[str(native.relative_to(REPO))] = digest(native)
    (HERE / "native_build.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Native dependencies prepared. Use launch.py inside the matching torch_musa image.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
