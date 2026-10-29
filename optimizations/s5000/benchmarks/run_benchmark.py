"""Run one formal round, with one excluded warm-up per input."""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path('/workspace/Protenix')
HERE = Path("/workspace/benchmarks")


def pair_inputs(inputs):
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("Input JSON must be a non-empty list")
    if any(not isinstance(x, dict) or not isinstance(x.get("name"), str) or not x["name"] for x in inputs):
        raise ValueError("Each input needs a non-empty string name")
    names = [x["name"] for x in inputs]
    if any(n in (".", "..") or "/" in n or "\\" in n for n in names):
        raise ValueError("Input names must be single safe path components")
    if len(set(names)) != len(names):
        raise ValueError("Input names must be unique")
    if any(n.endswith("_warm") for n in names):
        if len(inputs) % 2:
            raise ValueError("Paired input must have an even number of entries")
        for warm, formal in zip(inputs[::2], inputs[1::2]):
            if formal["name"].endswith("_warm") or warm["name"] != formal["name"] + "_warm":
                raise ValueError("Expected <name>_warm immediately before <name>")
            a, b = copy.deepcopy(warm), copy.deepcopy(formal)
            a.pop("name")
            b.pop("name")
            if a != b:
                raise ValueError("Warm and formal biological inputs must match")
        return copy.deepcopy(inputs)
    result = []
    for entry in inputs:
        warm = copy.deepcopy(entry)
        warm["name"] += "_warm"
        result.extend([warm, copy.deepcopy(entry)])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["test23", "eval21"])
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.set_defaults(samples=5, cycles=10, steps=200, seed=101, model_name="protenix_base_default_v0.5.0")
    args = parser.parse_args()
    args.input_json = Path("/workspace/data") / args.dataset / "input.json"
    if args.gpu not in range(8): parser.error("--gpu must be 0..7")
    for field in ("samples", "cycles", "steps"):
        if getattr(args, field) < 1:
            parser.error(f"--{field} must be positive")
    if not os.environ.get("PROTENIX_ROOT_DIR"):
        parser.error("Set PROTENIX_ROOT_DIR to your model/data root")
    required = [
        REPO / "protenix/model/layer_norm/fast_layer_norm_cuda_v2.so",
        REPO / "protenix/model/musa_kernels/libtri_attn_musa_tf32_bm128_w8.so",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        parser.error("Required native extension missing from the delivery:\n"
                     + "\n".join(missing))
    for name in ["library_reference.json", "best_practice_reference.json"]:
        if not (HERE / "checks" / name).is_file():
            parser.error("Missing validation file: " + str(HERE / "checks" / name))
    inputs = pair_inputs(json.loads(args.input_json.read_text(encoding="utf-8")))
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; choose a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    input_path = output / "input_warm_formal.json"
    input_path.write_text(json.dumps(inputs, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env.update(json.loads((HERE / "verified_flags.json").read_text()))
    env.pop("TORCH_ALLOW_TF32_MUBLAS_OVERRIDE", None)
    env.update(MUSA_VISIBLE_DEVICES=str(args.gpu), CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONDONTWRITEBYTECODE="1")
    env.update(S5000_RUN_DIR=str(output), LAYERNORM_TYPE="fast_layernorm", OMP_NUM_THREADS="4")
    env.setdefault("MUSA_VISIBLE_DEVICES", "0")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    paths = [REPO]
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths)) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    command = [
        sys.executable, "-B", str(HERE / "runtime_audit.py"),
        "--model_name", args.model_name, "--seeds", str(args.seed),
        "--input_json_path", str(input_path), "--dump_dir", str(output / "output"),
        "--model.N_cycle", str(args.cycles), "--sample_diffusion.N_sample", str(args.samples),
        "--sample_diffusion.N_step", str(args.steps),
        "--triangle_attention", "triattention", "--triangle_multiplicative", "torch",
        "--enable_diffusion_shared_vars_cache", "True",
        "--enable_efficient_fusion", "True", "--enable_tf32", "True",
    ]
    manifest = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": args.dataset, "physical_gpu": args.gpu, "process_device": 0,
        "input_json": str(args.input_json), "precision_policy": "Original AMP and explicit FP32 protections; TF32 on; no added FP32-to-BF16 shortcuts",
        "formal_rounds": 1, "formal_inputs": len(inputs) // 2, "warm_inputs": len(inputs) // 2,
        "samples": args.samples, "cycles": args.cycles, "steps": args.steps, "seed": args.seed,
        "command": command, "cache": True, "efficient_fusion": True, "tf32": True,
        "optimization_flags": {key: value for key, value in sorted(env.items())
                               if key.startswith(("PROTENIX_ENABLE_MUSA_", "PROTENIX_DISABLE_MUSA_"))},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (output / "run.log").open("w", encoding="utf-8") as log:
        status = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    (output / "exit_code.txt").write_text(str(status), encoding="utf-8")
    if status:
        print(f"Inference exited {status}; inspect {output / 'run.log'}", file=sys.stderr)
        return status
    summary = subprocess.run([sys.executable, "-B", str(HERE / "validate_run.py"), args.dataset, str(output)])
    return summary.returncode


if __name__ == "__main__":
    raise SystemExit(main())
