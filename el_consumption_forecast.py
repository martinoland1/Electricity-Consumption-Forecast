# run_and_extract_regression.py
# Käivitab regression_analysis_split.py ja parsib stdout'ist välja
# kahe segmendi (WORKDAYS ja WEEKENDS & HOLIDAYS) regressioonivõrrandid.
# Kõik salvestused on KOMMENTEERITUD.

import sys
import os
import re
import json
import subprocess
from pathlib import Path

WORKDAYS_HDR = r"=== WORKDAYS .*?Linear Regression Summary ==="
OFFDAYS_HDR = r"=== WEEKENDS\s*&\s*HOLIDAYS .*?Linear Regression Summary ==="

SLOPE_LINE = r"- Slope .*?:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
INTERCEPT_LINE = r"- Intercept.*?:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"


def extract_segment(block: str):
    slope_m = re.search(SLOPE_LINE, block)
    inter_m = re.search(INTERCEPT_LINE, block)
    if not slope_m or not inter_m:
        return None
    slope = float(slope_m.group(1))
    intercept = float(inter_m.group(1))
    return {"slope": slope, "intercept": intercept}


def extract_equations(stdout: str):
    # WORKDAYS plokk
    wd_block = None
    m = re.search(WORKDAYS_HDR + r"(.*?)(?:\n===|\Z)", stdout, flags=re.S)
    if m:
        wd_block = m.group(0)
    # WEEKENDS & HOLIDAYS plokk
    od_block = None
    m = re.search(OFFDAYS_HDR + r"(.*?)(?:\n===|\Z)", stdout, flags=re.S)
    if m:
        od_block = m.group(0)

    results = {}
    if wd_block:
        res = extract_segment(wd_block)
        if res:
            results["workdays"] = res
    if od_block:
        res = extract_segment(od_block)
        if res:
            results["offdays"] = res
    return results


def main():
    # Sihtskripti asukoht: vaikimisi sama kaust; võib ka käsureal anda
    if len(sys.argv) > 1:
        reg_path = Path(sys.argv[1]).resolve()
    else:
        reg_path = Path(__file__).with_name(
            "regression_analysis_split.py").resolve()

    if not reg_path.exists():
        print(f"ERROR: File not found: {reg_path}")
        sys.exit(1)

    # Väldi GUI blokki (matplotlib)
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"

    # Käivita regressiooniskript ja püüa stdout/stderr
    proc = subprocess.run(
        [sys.executable, str(reg_path)],
        cwd=str(reg_path.parent),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    if proc.returncode != 0:
        print(
            f"WARNING: regression script exited with code {proc.returncode}.")
        # # Toorlogide salvestamine (kommenteeritud):
        # out_file = reg_path.with_name("regression_stdout.txt")
        # err_file = reg_path.with_name("regression_stderr.txt")
        # out_file.write_text(proc.stdout or "", encoding="utf-8")
        # err_file.write_text(proc.stderr or "", encoding="utf-8")

    # Parsime võrrandid
    results = extract_equations(proc.stdout or "")

    if not results:
        print("No equations detected in stdout. Kontrolli, kas regression_analysis_split.py prindib kokkuvõtte samas formaadis.")
        sys.exit(2)

    # Inimloetav kokkuvõte (ainult print; salvestus välja kommenteeritud)
    if "workdays" in results:
        w = results["workdays"]
        print(f"WORKDAYS: y = {w['intercept']:.6f} + ({w['slope']:.6f})*x")
    else:
        print("WORKDAYS: not found")

    if "offdays" in results:
        o = results["offdays"]
        print(f"OFFDAYS : y = {o['intercept']:.6f} + ({o['slope']:.6f})*x")
    else:
        print("OFFDAYS : not found")

    # # JSON salvestus (kommenteeritud):
    # json_path = reg_path.with_name("segment_equations.json")
    # json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    # print(f"Saved: {json_path.name}")


if __name__ == "__main__":
    main()
