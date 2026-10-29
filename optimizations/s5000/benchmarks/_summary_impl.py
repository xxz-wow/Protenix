"""Validate generated structures and report warm-excluded timings."""
import argparse
import json
from pathlib import Path
import re


def summarize(run):
    from biotite.structure.io.pdbx import CIFFile, get_structure
    import numpy as np
    if (run / "exit_code.txt").read_text().strip() != "0":
        raise ValueError("Inference did not exit successfully")
    manifest = json.loads((run / "manifest.json").read_text())
    inputs = json.loads((run / "input_warm_formal.json").read_text())
    log = (run / "run.log").read_text()
    hits = re.findall(r"\] (\S+) \[seed:\d+\] succeeded\. Model forward time: ([\d.]+)s", log)
    times = {name: float(seconds) for name, seconds in hits}
    names = [entry["name"] for entry in inputs]
    if len(hits) != len(names) or len(times) != len(names) or set(times) != set(names):
        raise ValueError("Expected exactly one successful prediction for every warm and formal input")
    if "failed:" in log or "Traceback" in log:
        raise ValueError("Inference log contains errors")
    iterations = json.loads((run / "iteration_timings.json").read_text())
    if [row["name"] for row in iterations] != names:
        raise ValueError("Iteration windows do not match input order")
    cases = []
    samples = manifest["samples"]
    seed = manifest["seed"]
    for row in iterations:
        if row["data_error"]:
            raise ValueError(f"Data error: {row['name']}")
        folder = run / "output" / row["name"] / f"seed_{seed}" / "predictions"
        files = sorted(folder.glob("*.cif"))
        confidence = sorted(folder.glob("*summary_confidence*.json"))
        if len(files) != samples or len(confidence) != samples:
            raise ValueError(f"Unexpected sample count: {row['name']}")
        for sample in range(samples):
            expected = folder / f"{row['name']}_sample_{sample}.cif"
            atoms = get_structure(CIFFile.read(expected), model=1)
            if len(atoms) != row["shape"]["N_atom"] or not np.isfinite(atoms.coord).all():
                raise ValueError(f"Invalid coordinates: {expected}")
        for path in confidence:
            json.loads(path.read_text())
        if not row["warm"]:
            cases.append(dict(pdb_id=row["name"], shape=row["shape"], samples=samples,
                              log_seconds=times[row["name"]]))
    job_hits = re.findall(r"Job completed in ([\d.]+)s", log)
    if len(job_hits) != 1:
        raise ValueError("Expected one Job completed record")
    warm = sum(row["iteration_wall_seconds"] for row in iterations if row["warm"])
    result = {
        "formal_inputs": len(cases), "formal_structures": len(cases) * samples,
        "samples": samples, "formal_rounds": 1,
        "formal_log_seconds": sum(case["log_seconds"] for case in cases),
        "job_loop_including_warm_seconds": float(job_hits[0]),
        "warm_iteration_seconds_excluded": warm,
        "job_loop_excluding_warm_seconds": float(job_hits[0]) - warm,
        "cases": cases,
        "timing": "Formal logs include predict, postprocess and output; adjusted Job also includes "
                  "formal data preparation. Both exclude warm and model startup/loading.",
        "validation": "All warm and formal CIFs checked for sample/atom count and finite coordinates. "
                      "Structural accuracy equivalence has not been evaluated.",
    }
    (run / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# S5000 inference benchmark", "",
             "| Metric | Seconds |", "|---|---:|",
             f"| Formal input log sum | {result['formal_log_seconds']:.2f} |",
             f"| Job loop minus warm windows | {result['job_loop_excluding_warm_seconds']:.2f} |", "",
             f"Formal inputs: {len(cases)}; samples/input: {samples}; formal rounds: 1.",
             "", result["timing"], "", result["validation"], ""]
    (run / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = summarize(args.run_dir.resolve())
    print(json.dumps({key: result[key] for key in
          ("formal_inputs", "formal_structures", "formal_log_seconds", "job_loop_excluding_warm_seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
