#!/usr/bin/env python3
"""Run every exercise .py file, capture FULL output, and produce an execution report.

Artifacts written to _run_logs/:
  - <part>__<file>.log    full stdout+stderr transcript for each exercise
  - results.json          machine-readable status for every exercise
  - REPORT.md             human-readable summary report (also copied to repo root)

Also regenerates docs/REFERENCE_RESULTS.md (environment + headline metrics + sample
transcripts), so the published reference results never drift from the actual run.
"""
import os
import re
import sys
import subprocess
import time
import json
import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(ROOT, "_run_logs")
PART_PREFIXES = ("I.", "II.", "III.", "IV.", "V.", "VI.", "VII.", "VIII.", "IX.", "Appendices")
TIMEOUT = 180

# Headline metrics auto-extracted from each exercise's stdout for REFERENCE_RESULTS.md.
# (file_substring, part, description, regex). The regex captures one group; if it matches
# multiple times (e.g. two "Idle %" lines) the first→last values are shown.
HEADLINES = [
    ("3.3_hardware_survey",         "I",   "Roofline from device specs",            r"Peak FP16 FLOP/s\s*:\s*([\d.]+ TFLOP/s)"),
    ("4.2_execution_model",         "II",  "Throughput scales with batch (knee)",   r"Knee \(80% of peak\) at batch size:\s*(\d+)"),
    ("4.4_tensor_cores_and_fusion", "II",  "BF16 Tensor-Core speedup vs FP32",      r"BF16 speedup\s*:\s*([\d.]+x)\s*vs FP32"),
    ("6.1_precision_and_amp",       "II",  "autocast forward speedup",              r"Autocast forward\s*:.*\(([\d.]+x faster)\)"),
    ("7.1_dataloader_pipeline",     "II",  "removing in-loop .item()",              r"Speedup from removing \.item\(\)\s*:\s*([\d.]+x)"),
    ("7.2_io_bottleneck",           "II",  "GPU idle: starved → fed",          r"Idle %\s*:\s*([\d.]+%)"),
    ("9.1_memory_hierarchy",        "III", "sequential vs strided memory",          r"Slowdown \(row/col\):\s*([\d.]+×)"),
    ("10.2_prefill_and_decode",     "IV",  "TTFT grows with prompt length",         r"TTFT\[\d+\] / TTFT\[\d+\] = ([\d.]+×)"),
]

# Sample transcripts: slice each exercise's stdout from the line containing `start` to the
# line containing `end` (inclusive).
SHOWCASE = [
    ("3.3_hardware_survey",  "GPU name",         "compute bound (Tensor Cores useful)"),
    ("4.2_execution_model",  "batch=   1",       "Knee (80% of peak)"),
    ("9.1_memory_hierarchy", "Row-major copy",   "Slowdown (row/col)"),
]

def find_exercises():
    files = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel = os.path.relpath(dirpath, ROOT)
        top = rel.split(os.sep)[0]
        if top in ("shared", ".git", "_run_logs"):
            continue
        if not (top.startswith(PART_PREFIXES) or rel == "."):
            continue
        for fn in filenames:
            if fn.endswith(".py") and fn != os.path.basename(__file__):
                files.append(os.path.join(dirpath, fn))
    return sorted(files)

def classify(rc, out, err, timed_out):
    low = (out + "\n" + err).lower()
    if timed_out:
        return "TIMEOUT"
    if rc == 0:
        return "PASS"
    if "notimplementederror" in low:
        return "INCOMPLETE"
    if "modulenotfounderror" in low or "importerror" in low or "dll load failed" in low:
        return "IMPORT_ERROR"
    if "assertionerror" in low:
        return "ASSERTION_FAIL"
    return "ERROR"

def last_error(out, err):
    text = err.strip() or out.strip()
    lines = [l for l in text.splitlines() if l.strip()]
    for l in reversed(lines):
        if "Error" in l or "Exception" in l:
            return l.strip()[:300]
    return (lines[-1].strip()[:300] if lines else "")


def environment():
    """Best-effort capture of the GPU / framework environment."""
    info = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["vram_gb"] = round(p.total_memory / 1e9, 1)
            info["sms"] = p.multi_processor_count
            info["cc"] = f"{p.major}.{p.minor}"
    except Exception:
        pass
    try:
        drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if drv.returncode == 0 and drv.stdout.strip():
            info["driver"] = drv.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return info


def extract_headline(stdout, pattern):
    """Return the captured value (or first→last if a pattern matches several lines)."""
    matches = re.findall(pattern, stdout or "")
    if not matches:
        return "—"
    if len(matches) == 1:
        return matches[0]
    return f"{matches[0]} → {matches[-1]}"


def slice_transcript(stdout, start, end, max_lines=14):
    """Return the stdout lines from `start` to `end` (inclusive), de-indented."""
    lines = (stdout or "").splitlines()
    s = next((i for i, l in enumerate(lines) if start in l), None)
    if s is None:
        return ""
    e = next((j for j in range(s, len(lines)) if end in lines[j]), s)
    chunk = lines[s:e + 1][:max_lines]
    return "\n".join(l.strip() for l in chunk)


def write_reference_results(results, captured):
    """Regenerate docs/REFERENCE_RESULTS.md from the actual run (data-driven sections)."""
    info = environment()
    by_file = {r["file"]: r for r in results}
    npass = sum(1 for r in results if r["status"] == "PASS")
    n_exercise = sum(1 for r in results if os.path.basename(r["file"]) != "sitecustomize.py")
    npass_ex = sum(1 for r in results
                   if r["status"] == "PASS" and os.path.basename(r["file"]) != "sitecustomize.py")
    total_s = round(sum(r["dur"] for r in results), 1)
    slowest = sorted(results, key=lambda r: -r["dur"])[:3]
    date = datetime.datetime.now().strftime("%Y-%m-%d")

    L = []
    L.append("# Reference Results")
    L.append("")
    L.append("> **Auto-generated by `run_all_exercises.py` — do not edit by hand.** "
             "Re-run the suite to refresh.")
    L.append("")
    L.append("Sample results from a reference run of every exercise. Use them to confirm your own "
             "setup is healthy and to see what correct output looks like.")
    L.append("")
    L.append("> ⚠️ **Absolute numbers are hardware- and version-specific.** The **ratios and the "
             "direction** of each effect are the lesson, not the exact ms/GB/s. A few timing-based "
             "checks (4.2, 7.1, 7.2, 10.2) also vary run-to-run.")
    L.append("")
    L.append("Full machine-readable report: [`../EXERCISE_EXECUTION_REPORT.md`]"
             "(../EXERCISE_EXECUTION_REPORT.md) · per-file logs in [`../_run_logs/`](../_run_logs/).")
    L.append("")
    L.append("## Reference Environment")
    L.append("")
    L.append("| | |")
    L.append("|---|---|")
    if info.get("gpu"):
        L.append(f"| **GPU** | {info['gpu']} "
                 f"({info.get('vram_gb','?')} GB, {info.get('sms','?')} SMs, "
                 f"Compute Capability {info.get('cc','?')}) |")
    if info.get("driver"):
        L.append(f"| **Driver** | {info['driver']} |")
    L.append(f"| **PyTorch / CUDA** | {info.get('torch','?')} / {info.get('cuda','?')} |")
    L.append(f"| **Python** | {sys.version.split()[0]} |")
    L.append(f"| **Run date** | {date} |")
    L.append("")
    L.append("## Summary")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| **Exercises passing** | **{npass_ex} / {n_exercise}** "
             f"(harness runs {npass}/{len(results)} incl. the `sitecustomize.py` bootstrap) |")
    L.append(f"| Failures | {len(results) - npass} |")
    L.append(f"| Total wall time | ~{total_s:.0f} s |")
    L.append("| Slowest exercises | "
             + ", ".join(f"`{os.path.basename(r['file'])}` ({r['dur']:.1f} s)" for r in slowest)
             + " |")
    L.append("")
    L.append("## Headline Results")
    L.append("")
    L.append("Auto-extracted from each exercise's output. Illustrative — expect different absolute "
             "values on other hardware.")
    L.append("")
    L.append("| Part | Exercise | Demonstrates | Result |")
    L.append("|---|---|---|---|")
    for sub, part, desc, pat in HEADLINES:
        match = next((f for f in captured if sub in f), None)
        value = extract_headline(captured.get(match, ""), pat) if match else "—"
        name = os.path.basename(match) if match else sub
        L.append(f"| {part} | `{name}` | {desc} | {value} |")
    L.append("")
    L.append("> Parts V–IX (benchmarking methodology, capstone labs, hardware landscape, advanced "
             "topics, production capstones) are methodology / portfolio / read-and-run material; they "
             "execute cleanly and emit before/after comparison tables rather than a single headline number.")
    L.append("")
    L.append("## Sample Transcripts")
    L.append("")
    L.append("Trimmed stdout from a few flagship exercises (full transcripts in `_run_logs/`).")
    L.append("")
    for sub, start, end in SHOWCASE:
        match = next((f for f in captured if sub in f), None)
        if not match:
            continue
        snippet = slice_transcript(captured.get(match, ""), start, end)
        if not snippet:
            continue
        L.append(f"### `{match}`")
        L.append("")
        L.append("```")
        L.append(snippet)
        L.append("```")
        L.append("")
    L.append("## Reproducing These Results")
    L.append("")
    L.append("```bash")
    L.append("python run_all_exercises.py")
    L.append("```")
    L.append("")
    L.append("This re-runs every exercise, refreshes `_run_logs/`, and regenerates both "
             "`EXERCISE_EXECUTION_REPORT.md` and this file.")
    L.append("")

    docsdir = os.path.join(ROOT, "docs")
    os.makedirs(docsdir, exist_ok=True)
    with open(os.path.join(docsdir, "REFERENCE_RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))

def main():
    os.makedirs(LOGDIR, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    files = find_exercises()
    results = []
    captured = {}
    for i, path in enumerate(files, 1):
        rel = os.path.relpath(path, ROOT)
        cwd = os.path.dirname(path)
        sys.stderr.write(f"[{i}/{len(files)}] {rel}\n")
        sys.stderr.flush()
        t0 = time.time()
        timed_out = False
        try:
            p = subprocess.run([sys.executable, os.path.basename(path)],
                               cwd=cwd, env=env, capture_output=True, text=True,
                               timeout=TIMEOUT)
            rc, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            rc = -1
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            timed_out = True
        dur = time.time() - t0
        status = classify(rc, out, err, timed_out)
        logname = rel.replace("\\", "__").replace("/", "__").replace(".py", "") + ".log"
        logpath = os.path.join(LOGDIR, logname)
        with open(logpath, "w", encoding="utf-8") as f:
            f.write(f"# {rel}\n# status={status} rc={rc} duration={dur:.1f}s\n")
            f.write("=" * 70 + "\n--- STDOUT ---\n")
            f.write(out)
            if err.strip():
                f.write("\n--- STDERR ---\n")
                f.write(err)
        results.append({
            "file": rel.replace("\\", "/"),
            "status": status,
            "rc": rc,
            "dur": round(dur, 1),
            "log": "_run_logs/" + logname,
            "stdout_lines": len(out.splitlines()),
            "error": "" if status == "PASS" else last_error(out, err),
        })
        captured[rel.replace("\\", "/")] = out

    with open(os.path.join(LOGDIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # --- markdown report ---
    from collections import Counter
    c = Counter(r["status"] for r in results)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    md = []
    md.append("# Exercise Execution Report")
    md.append("")
    md.append(f"_Generated {now} · Python {sys.version.split()[0]} · {len(results)} exercises_")
    md.append("")
    md.append("## Summary")
    md.append("")
    md.append("| Status | Count |")
    md.append("|---|---|")
    for k, v in c.most_common():
        md.append(f"| {k} | {v} |")
    md.append(f"| **TOTAL** | **{len(results)}** |")
    md.append("")
    md.append("## Per-exercise results")
    md.append("")
    md.append("| # | Exercise | Status | Time (s) | Detail |")
    md.append("|---|---|---|---|---|")
    for i, r in enumerate(results, 1):
        detail = r["error"].replace("|", "\\|") if r["error"] else f'{r["stdout_lines"]} lines out'
        md.append(f'| {i} | {r["file"]} | {r["status"]} | {r["dur"]} | {detail} |')
    md.append("")
    report = "\n".join(md)
    with open(os.path.join(LOGDIR, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(ROOT, "EXERCISE_EXECUTION_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report)

    # regenerate docs/REFERENCE_RESULTS.md from this run (never drifts)
    try:
        write_reference_results(results, captured)
    except Exception as e:
        sys.stderr.write(f"WARNING: could not write REFERENCE_RESULTS.md: {e}\n")

    sys.stderr.write("\n=== SUMMARY ===\n")
    for k, v in c.most_common():
        sys.stderr.write(f"{k}: {v}\n")
    sys.stderr.write(f"TOTAL: {len(results)}\n")
    sys.stderr.write(f"Logs in {LOGDIR}\n")

if __name__ == "__main__":
    main()
