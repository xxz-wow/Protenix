"""Run one formal round, with one excluded warm-up per input."""
import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


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
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--model-name", default="protenix_base_default_v0.5.0")
    args = parser.parse_args()
    for field in ("samples", "cycles", "steps"):
        if getattr(args, field) < 1:
            parser.error(f"--{field} must be positive")
    if not os.environ.get("PROTENIX_ROOT_DIR"):
        parser.error("Set PROTENIX_ROOT_DIR to your model/data root")
    required = [
        REPO / "protenix/model/layer_norm/fast_layer_norm_cuda_v2.so",
        REPO / "protenix/model/musa_kernels/mutlass_triangle_bmm/protenix_mutlass_triangle_bmm_v5.so",
        REPO / "protenix/model/tri_attention/v21/libv21_training_fast_p36q32_repro.so",
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        parser.error("Prepare native dependencies first: python optimizations/s5000/prepare_native.py\n"
                     + "\n".join(missing))
    inputs = pair_inputs(json.loads(args.input_json.read_text(encoding="utf-8")))
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; choose a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    input_path = output / "input_warm_formal.json"
    input_path.write_text(json.dumps(inputs, indent=2), encoding="utf-8")
    env = os.environ.copy()
    env.update(S5000_RUN_DIR=str(output), LAYERNORM_TYPE="torch", OMP_NUM_THREADS="4",
               XXZ_MUTLASS_VALIDATE="0")
    env.setdefault("MUSA_VISIBLE_DEVICES", "0")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    paths = [REPO]
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths)) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    command = [
        sys.executable, "-B", str(REPO / "runner/inference.py"),
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
        "formal_rounds": 1, "formal_inputs": len(inputs) // 2, "warm_inputs": len(inputs) // 2,
        "samples": args.samples, "cycles": args.cycles, "steps": args.steps, "seed": args.seed,
        "command": command, "cache": True, "efficient_fusion": True, "tf32": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (output / "run.log").open("w", encoding="utf-8") as log:
        status = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    (output / "exit_code.txt").write_text(str(status), encoding="utf-8")
    if status:
        print(f"Inference exited {status}; inspect {output / 'run.log'}", file=sys.stderr)
        return status
    summary = subprocess.run([sys.executable, "-B", str(HERE / "summarize.py"), str(output)])
    return summary.returncode


if __name__ == "__main__":
    raise SystemExit(main())
