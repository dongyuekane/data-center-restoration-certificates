"""Verify the saved confirmation records and summarize the complete costs."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from math import isclose
from pathlib import Path
from statistics import median


LABELS = {
    "warm_lp": "Persistent LP",
    "nearest_certificate": "Nearest-certificate cache",
    "all_certificate": "Certificate bank",
    "monotone_bracket": "Monotone bracket",
}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify_and_summarize(results):
    root = Path(__file__).resolve().parent
    protocol = json.loads((root / "protocol.json").read_text())
    profiles = json.loads((results / "confirmation_profiles.json").read_text())
    audit = json.loads((results / "audit_metrics.json").read_text())
    measured = json.loads((results / "confirmation_metrics.json").read_text())
    assembly = json.loads((results / "model_cost.json").read_text())
    queries = read_csv(results / "confirmation_queries.csv")
    timing = read_csv(results / "confirmation_timings.csv")
    construction = read_csv(results / "model_construction.csv")
    model_keys = {(str(p["id"]), str(w), str(d)) for p in profiles for w in protocol["wait_minutes"] for d in protocol["downtime_minutes"]}
    queries_per_model = protocol["queries_per_model"]
    expected_queries = len(model_keys) * queries_per_model
    if (len(profiles), len(queries), measured["queries"], audit["queries"]) != (
        protocol["confirmation_profiles"], expected_queries, expected_queries, expected_queries
    ):
        raise ValueError("Profile or confirmation query counts differ from the frozen protocol.")

    model_counts = Counter((r["id"], r["wait"], r["lag"]) for r in queries)
    query_keys = {(r["id"], r["wait"], r["lag"], r["query"]) for r in queries}
    if len(query_keys) != expected_queries or model_counts != dict.fromkeys(model_keys, queries_per_model):
        raise ValueError("Duplicate or missing confirmation queries.")

    for row in queries:
        loss, budget = float(row["oracle_loss"]), float(row["budget"])
        if (row["feasible"] == "True") != (loss <= budget + 1e-8):
            raise ValueError("A query decision differs from the reference LP objective.")
        for method in LABELS.keys() - {"warm_lp"}:
            lower, upper = float(row[f"{method}_lower"]), float(row[f"{method}_upper"])
            if not lower <= loss + 1e-6 or not upper >= loss - 1e-6:
                raise ValueError(f"{method} bound does not contain a reference loss.")

    expected_boundary = len(model_keys) * len(range(0, queries_per_model, 8)) * 2
    if (audit["near_boundary_queries"] != expected_boundary or audit["near_boundary_mismatches"]
            or audit["max_repaired_primal_violation"] >= 1e-7 or measured["mismatches"]):
        raise ValueError("The boundary, witness or decision audit failed.")

    setup_by_model = defaultdict(list)
    for row in construction:
        setup_by_model[row["id"], row["wait"], row["lag"]].append(float(row["seconds"]))
    setup = sum(median(s) for s in setup_by_model.values())
    if (set(setup_by_model) != model_keys or any(len(rows) != 5 for rows in setup_by_model.values())
            or not isclose(setup, assembly["total_s"], abs_tol=1e-12)):
        raise ValueError("Model construction samples do not match the saved summary.")

    totals = {}
    for method, label in LABELS.items():
        sample = defaultdict(list)
        for row in timing:
            if row["method"] != method:
                continue
            if int(row["mismatch"]) or int(row["queries"]) != queries_per_model:
                raise ValueError(f"Invalid timing record for {method}.")
            sample[row["id"], row["wait"], row["lag"]].append(float(row["total_s"]))
        elapsed = sum(median(times) for times in sample.values())
        expected = measured["methods"][method]
        if (set(sample) != model_keys or any(len(times) != protocol["repetitions"] for times in sample.values())
                or not isclose(elapsed, expected["total_s"], abs_tol=1e-12)):
            raise ValueError(f"Timing samples do not match the saved summary for {method}.")
        if method != "warm_lp":
            observed = sum(r[f"{method}_fallback"] == "True" for r in queries)
            if observed != expected["fallback"]:
                raise ValueError(f"Query and timing fallbacks differ for {method}.")
        totals[method] = dict(label=label, total_s=elapsed + setup, fallbacks=expected["fallback"])

    return dict(profiles=len(profiles), models=len(model_keys), queries=len(queries), boundary_queries=expected_boundary,
                boundary_fallbacks=audit["near_boundary_fallback"], mismatches=0, common_construction_s=setup, methods=totals)


def plot(summary, destination):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    methods = list(summary["methods"].values())
    labels = [record["label"] for record in methods]
    times = [1000 * record["total_s"] for record in methods]
    fig, ax = plt.subplots(figsize=(7.2, 2.8), layout="constrained")
    bars = ax.barh(labels, times, height=0.58, color=["#748392", "#748392", "#167565", "#748392"])
    ax.bar_label(bars, fmt="%.1f", padding=5)
    ax.invert_yaxis()
    ax.set_xlabel("Complete confirmation time (ms)")
    ax.set_xlim(0, max(times) * 1.14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e2e8ec", linewidth=0.7)
    ax.set_axisbelow(True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "archive")
    parser.add_argument("--figure", type=Path, help="Optional output path for a timing chart")
    args = parser.parse_args()
    summary = verify_and_summarize(args.results)
    print(json.dumps(summary, indent=2))
    if args.figure:
        plot(summary, args.figure)
