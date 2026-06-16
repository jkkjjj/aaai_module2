#!/usr/bin/env python3
"""Scan evaluation summaries and compute AUC = mean(individual_scores) / Rmax."""
import csv
import json
import os
import re
from pathlib import Path

GAMES = ["detective", "library", "zork1", "zork3", "balances", "temple", "deephome", "ztuu", "ludicorp", "pentari"]
METHODS = [ "our"]
RMAX = {
    "detective": 360,
    "library": 30,
    "zork1": 350,
    "zork3": 7,
    "balances": 51,
    "temple": 35,

    # 下面 4 个论文主文 Figure 2 没给，需要用 Jericho env.get_max_score() 或日志确认
    "deephome": 35,
    "ztuu": 100,
    "ludicorp": 20,
    "pentari": 35,
}
MODEL = "gpt-oss-20b"
ROOT = Path(__file__).resolve().parent

TIMESTAMP_RE = re.compile(r"(\d{8}-\d{6})")


def extract_timestamp(path: Path):
    """Pull timestamp from any path component, e.g. 20260611-003347."""
    for part in path.parts:
        m = TIMESTAMP_RE.search(part)
        if m:
            return m.group(1)
    return None


def find_summary(game: str, method: str):
    """Return latest summary JSON path containing 'individual_scores', or None."""
    base = ROOT / game / method / MODEL
    if not base.is_dir():
        return None

    candidates = []
    for json_path in base.rglob("*.json"):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict) or "individual_scores" not in data:
            continue
        candidates.append(json_path)

    if not candidates:
        return None

    def sort_key(p: Path):
        ts = extract_timestamp(p)
        if ts is not None:
            return (0, ts)
        return (1, str(p.stat().st_mtime))

    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]


def load_scores(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("individual_scores", [])


def fmt_scores(scores):
    return "[" + ",".join(str(s) for s in scores) + "]"


def main():
    rows = []
    for game in GAMES:
        rmax = RMAX[game]
        for method in METHODS:
            summary_path = find_summary(game, method)
            if summary_path is None:
                rows.append({
                    "game": game,
                    "method": method,
                    "summary_file": "",
                    "n_episodes": "NA",
                    "scores": "NA",
                    "avg_score": "NA",
                    "rmax": rmax,
                    "auc": "NA",
                })
                continue

            scores = load_scores(summary_path)
            if not scores:
                rows.append({
                    "game": game,
                    "method": method,
                    "summary_file": str(summary_path.relative_to(ROOT)),
                    "n_episodes": 0,
                    "scores": "NA",
                    "avg_score": "NA",
                    "rmax": rmax,
                    "auc": "NA",
                })
                continue

            avg = sum(scores) / len(scores)
            auc = avg / rmax
            rows.append({
                "game": game,
                "method": method,
                "summary_file": str(summary_path.relative_to(ROOT)),
                "n_episodes": len(scores),
                "scores": fmt_scores(scores),
                "avg_score": f"{avg:.4f}",
                "rmax": rmax,
                "auc": f"{auc:.4f}",
            })

    csv_path = ROOT / "auc_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "game", "method", "summary_file", "n_episodes",
                "scores", "avg_score", "rmax", "auc",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "| game | method | scores | avg_score | rmax | auc |",
        "|------|--------|--------|-----------|------|-----|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['game']} | {r['method']} | {r['scores']} | "
            f"{r['avg_score']} | {r['rmax']} | {r['auc']} |"
        )
    md_text = "\n".join(md_lines) + "\n"

    md_path = ROOT / "auc_results.md"
    md_path.write_text(md_text, encoding="utf-8")

    print(md_text)
    print(f"\nSaved CSV : {csv_path}")
    print(f"Saved MD  : {md_path}")


if __name__ == "__main__":
    main()
