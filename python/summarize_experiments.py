"""Export all completed mission reports, including failed experiments."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    for path in sorted((ROOT / "logs").glob("mission_*/report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"run": path.parent.name, "scenario": report["scenario"],
                     "status": report["status"], "hit_vehicle": report.get("hit_evidence", {}).get("vehicle", ""),
                     "landing_verified": report.get("landing_verified", False),
                     "min_separation_m": report.get("min_separation_m"),
                     "max_waiting_drift_m": report.get("max_waiting_drift_m"),
                     "miss_to_dispatch_s": report.get("miss_to_dispatch_s"),
                     "clear_to_dispatch_s": report.get("clear_to_dispatch_s"),
                     "error": report.get("error", "")})
    output = ROOT / "logs" / "experiment_summary.csv"
    if rows:
        with output.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"Summary: {output}")


if __name__ == "__main__":
    main()
