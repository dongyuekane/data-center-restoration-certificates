"""Run the frozen comparison on generated recovery profiles."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from restoration import Bank, Oracle, P, PROJECT_ROOT, PROTOCOL_PATH, model, np, profile


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def prepare_output(path):
    output = path.resolve()
    archive = PROJECT_ROOT / "archive"
    if output == PROJECT_ROOT or output.is_relative_to(archive):
        raise ValueError("Choose a generated output directory outside the archived results.")
    output.mkdir(parents=True, exist_ok=True)
    return output


def get_bounds(bank, method, energy, migration_cost):
    if method == "monotone_bracket":
        return bank.monotone(energy, migration_cost)
    return bank.bounds(energy, migration_cost, method)


def time_method(lp_model, method, parameters, reference):
    start = perf_counter()
    oracle = Oracle(lp_model)
    bank = None if method == "warm_lp" else Bank(lp_model, oracle)
    preparation = perf_counter() - start
    decisions = []
    fallback = 0
    for energy, cost, budget in parameters:
        if bank is None:
            decisions.append(oracle.solve(energy, cost) <= budget + 1e-8)
            continue
        lower, upper = get_bounds(bank, method, energy, cost)
        if upper <= budget:
            decisions.append(True)
        elif lower > budget:
            decisions.append(False)
        else:
            fallback += 1
            decisions.append(oracle.solve(energy, cost) <= budget + 1e-8)
    elapsed = perf_counter() - start
    mismatches = int(np.count_nonzero(np.asarray(decisions) != reference))
    if mismatches:
        raise AssertionError(f"{method} disagreed with the full LP on {mismatches} queries")
    return dict(total_s=elapsed, pre_s=preparation, fallback=fallback, mismatch=mismatches)


def run(split, profile_count, queries_per_model, repetitions, seed, output):
    rng = np.random.default_rng(seed)
    profiles = [profile(rng, index) for index in range(profile_count)]
    (output / f"{split}_profiles.json").write_text(json.dumps(profiles, indent=2))
    query_rows, timings = [], []
    for recovery in profiles:
        profile_id = recovery["id"]
        for wait in P["wait_minutes"]:
            for downtime in P["downtime_minutes"]:
                lp_model = model(recovery, wait, downtime)
                energy = rng.uniform(2, 10, queries_per_model)
                migration_cost = rng.uniform(0, 4, queries_per_model)
                budgets = rng.uniform(0, 10, queries_per_model)
                parameters = list(zip(energy, migration_cost, budgets))
                oracle = Oracle(lp_model)
                truth = np.array([oracle.solve(e, k) for e, k, _ in parameters])
                reference = truth <= budgets + 1e-8
                # A fresh solver matches the bank's deployment preparation.
                bank = Bank(lp_model, Oracle(lp_model))
                context = dict(split=split, id=profile_id, scenarios=len(recovery["grid"]), wait=wait, lag=downtime)
                for index, ((e, k, budget), loss) in enumerate(zip(parameters, truth)):
                    bounds = {}
                    for method in P["methods"][1:]:
                        lower, upper = get_bounds(bank, method, e, k)
                        if not lower <= loss + 1e-6 or not upper >= loss - 1e-6:
                            raise AssertionError(f"Invalid archived-format bound for {context}, query {index}")
                        bounds[f"{method}_lower"] = lower
                        bounds[f"{method}_upper"] = upper
                        bounds[f"{method}_fallback"] = bool(lower <= budget < upper)
                    query_rows.append(dict(**context, query=index, E=e, kappa=k, budget=budget, oracle_loss=loss,
                                           feasible=bool(reference[index]), **bounds))
                for repeat in range(repetitions):
                    order = np.random.default_rng(seed + profile_id * 20 + wait + downtime + repeat)
                    for method in order.permutation(P["methods"]):
                        timing = time_method(lp_model, method, parameters, reference)
                        timings.append(dict(**context, rep=repeat, method=method, queries=len(parameters), **timing))
        print(f"{split} profile {profile_id + 1}/{profile_count}", flush=True)

    write_csv(output / f"{split}_queries.csv", query_rows)
    write_csv(output / f"{split}_timings.csv", timings)
    methods = {}
    for method in P["methods"]:
        rows = [row for row in timings if row["method"] == method]
        profile_s = []
        preparation_s = []
        for profile_id in range(profile_count):
            total, preparation = 0.0, 0.0
            for wait in P["wait_minutes"]:
                for downtime in P["downtime_minutes"]:
                    model_rows = [row for row in rows if (row["id"], row["wait"], row["lag"]) == (profile_id, wait, downtime)]
                    total += median(row["total_s"] for row in model_rows)
                    preparation += median(row["pre_s"] for row in model_rows)
            profile_s.append(total)
            preparation_s.append(preparation)
        methods[method] = dict(total_s=sum(profile_s), profile_s=profile_s, pre_s=sum(preparation_s),
                               fallback=sum(row["fallback"] for row in rows if row["rep"] == 0))
    result = dict(profiles=profile_count, queries=len(query_rows), methods=methods, mismatches=0)
    (output / f"{split}_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["smoke", "development", "confirmation"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = args.split
    frozen = json.loads((PROJECT_ROOT / "frozen_selection.json").read_text())
    protocol_hash = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    if protocol_hash != frozen["protocol_sha256"]:
        raise ValueError("Protocol does not match the frozen method selection.")

    source_split = "confirmation" if split == "smoke" else split
    smoke = split == "smoke"
    result = run(split, 2 if smoke else P[f"{split}_profiles"], 8 if smoke else P["queries_per_model"],
                 1 if smoke else P["repetitions"], P[f"seed_{source_split}"], prepare_output(args.output))
    print(f"{result['queries']} queries, {result['mismatches']} disagreements with the full LP")


if __name__ == "__main__":
    main()
