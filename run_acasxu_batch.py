import configparser
import csv
import shutil
import subprocess
import time
from pathlib import Path

PROJECT = Path(".").resolve()
CONFIG_PATH = PROJECT / "nn-config.ini"
TEMP_CONFIG = PROJECT / "nn-config-temp.ini"

ONNX_DIR = PROJECT / "acasxu_2023" / "onnx"
VNNLIB_DIR = PROJECT / "acasxu_2023" / "vnnlib"

RESULTS_DIR = PROJECT / "acas_results"
RESULTS_DIR.mkdir(exist_ok=True)

TIMEOUT_SECONDS = 60

SPECIAL_NETS = {
    f"ACASXU_run2a_1_{i}_batch_2000.onnx"
    for i in range(1, 10)
}

def write_temp_config(network_path: Path, prop_path: Path):
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG_PATH)

    cfg["settings"]["nnpath"] = f'"{network_path.as_posix()}"'
    cfg["settings"]["specpath"] = f'"{prop_path.as_posix()}"'

    with open(TEMP_CONFIG, "w") as f:
        cfg.write(f)

def parse_data_csv():
    data_path = PROJECT / "data.csv"
    if not data_path.exists():
        return "", ""

    intervals = []
    runtime = ""

    with open(data_path, newline="") as f:
        rows = list(csv.reader(f))

    # Expected pandas CSV format:
    # ,0Low,0High
    # 0,low0,high0
    # 1,low1,high1
    # ...
    # last_row,runtime,runtime

    body = rows[1:]
    if not body:
        return "", ""

    for row in body[:-1]:
        idx, low, high = row
        intervals.append(f"Y{idx}: [{low}, {high}]")

    runtime = body[-1][1]

    return "; ".join(intervals), runtime

def detect_status(log_text: str, timed_out: bool):
    if timed_out:
        return "TIMEOUT"
    if "Safety Status: Safe" in log_text:
        return "UNSAT/SAFE"
    if "Safety Status: Unsafe" in log_text:
        return "SAT/UNSAFE"
    return "UNKNOWN"

networks = sorted(ONNX_DIR.glob("*.onnx"))

props_1_to_4 = [VNNLIB_DIR / f"prop_{i}.vnnlib" for i in range(1, 5)]
props_5_to_10 = [VNNLIB_DIR / f"prop_{i}.vnnlib" for i in range(5, 11)]

summary_path = RESULTS_DIR / "summary.csv"

with open(summary_path, "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow([
        "network",
        "property",
        "result",
        "runtime",
        "output_intervals",
        "observation",
    ])

    for network in networks:
        properties = list(props_1_to_4)

        if network.name in SPECIAL_NETS:
            properties += props_5_to_10

        for prop in properties:
            print(f"Running {network.name} with {prop.name}")

            # Remove old outputs so stale results cannot be mistaken for new ones.
            for old_file in ["data.csv", "log.txt"]:
                old_path = PROJECT / old_file
                if old_path.exists():
                    old_path.unlink()

            write_temp_config(network, prop)

            start = time.time()

            try:
                result = subprocess.run(
                    ["python", "nn-averinn.py", str(TEMP_CONFIG)],
                    cwd=PROJECT,
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT_SECONDS,
                )
                timed_out = False
                exit_code = result.returncode
                stderr = result.stderr
            except subprocess.TimeoutExpired as e:
                timed_out = True
                exit_code = 124
                stderr = e.stderr or ""

            elapsed = round(time.time() - start, 2)

            log_path = PROJECT / "log.txt"
            log_text = log_path.read_text(errors="ignore") if log_path.exists() else ""

            status = detect_status(log_text, timed_out)
            intervals, runtime = parse_data_csv()

            if not runtime:
                runtime = elapsed

            observation = ""
            if exit_code not in (0, 124):
                observation += f"exit_code={exit_code}; "
            if not intervals:
                observation += "no data.csv intervals; "
            if stderr:
                observation += str(stderr).replace("\n", " ")[:500]

            run_name = f"{network.stem}_{prop.stem}"

            if log_path.exists():
                shutil.copy(log_path, RESULTS_DIR / f"{run_name}.log")

            data_path = PROJECT / "data.csv"
            if data_path.exists():
                shutil.copy(data_path, RESULTS_DIR / f"{run_name}.csv")

            writer.writerow([
                network.name,
                prop.name,
                status,
                runtime,
                intervals,
                observation,
            ])

            out.flush()

print(f"Done. Results saved to {summary_path}")