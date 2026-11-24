"""Check boundary decisions, repair feasibility and model assembly cost."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from time import perf_counter

from experiment import prepare_output, write_csv
from restoration import Bank, Oracle, P, model, np


def audit(output):
    profiles = json.loads((output / "confirmation_profiles.json").read_text())
    groups = defaultdict(list)
    with (output / "confirmation_queries.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            groups[tuple(int(row[key]) for key in ("id", "wait", "lag"))].append(row)

    impact = []
    near_boundary = fallbacks = mismatches = 0
    maximum_violation = 0.0
    for recovery in profiles:
        profile_id = recovery["id"]
        for wait in P["wait_minutes"]:
            ideal = Oracle(model(recovery, wait, 0))
            for downtime in P["downtime_minutes"]:
                lp_model = model(recovery, wait, downtime)
                oracle = Oracle(lp_model)
                bank = Bank(lp_model, oracle)
                base, parameter, rhs_template, caps, _, energy_rows = lp_model
                for index, row in enumerate(groups[profile_id, wait, downtime]):
                    energy, migration, reference, budget = (float(row[k]) for k in ("E", "kappa", "oracle_loss", "budget"))
                    ideal_loss = ideal.solve(energy, 0)
                    impact.append(dict(id=profile_id, wait=wait, lag=downtime, E=energy, kappa=migration,
                                       ideal_loss=ideal_loss, physical_loss=reference, underestimate=reference - ideal_loss,
                                       ideal_admits=ideal_loss <= budget + 1e-8, physical_admits=reference <= budget + 1e-8))
                    if index % 8:
                        continue

                    lower, upper = bank.bounds(energy, migration, "all_certificate")
                    for delta in (-1e-5, 1e-5):
                        threshold = reference + delta
                        fallback = lower <= threshold < upper
                        estimate = oracle.solve(energy, migration) <= threshold + 1e-8 if fallback else upper <= threshold
                        near_boundary += 1
                        fallbacks += fallback
                        mismatches += estimate != (reference <= threshold + 1e-8)

                    # Re-evaluate each repaired vector directly in the queried LP.
                    rhs = rhs_template.copy()
                    rhs[energy_rows] = energy
                    matrix = base + migration * parameter
                    for witness in bank.X:
                        used = matrix @ witness
                        positive = bank.resource & (used > 0)
                        scale = min(1, float(np.min(rhs[positive] / used[positive]))) * (1 - 1e-10)
                        repaired = witness * scale
                        service = (base @ repaired)[bank.service] + repaired[-1]
                        repaired[-1] = max(0, float(np.max(20 + service))) + 1e-8
                        maximum_violation = max(maximum_violation, float(np.max(matrix @ repaired - rhs)),
                                                float(np.max(repaired - caps)), float(-repaired.min()))

    write_csv(output / "physical_impact.csv", impact)
    report = dict(queries=len(impact), ideal_model_false_admissions=sum(r["ideal_admits"] and not r["physical_admits"] for r in impact),
                  ideal_model_admissions=sum(r["ideal_admits"] for r in impact),
                  mean_loss_underestimate=float(np.mean([r["underestimate"] for r in impact])),
                  max_loss_underestimate=max(r["underestimate"] for r in impact), near_boundary_queries=near_boundary,
                  near_boundary_fallback=fallbacks, near_boundary_mismatches=mismatches,
                  max_repaired_primal_violation=maximum_violation)
    if mismatches or maximum_violation >= 1e-7:
        raise AssertionError(f"Boundary or witness audit failed: {report}")
    (output / "audit_metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    assembly = []
    for repeat in range(5):
        for recovery in profiles:
            for wait in P["wait_minutes"]:
                for downtime in P["downtime_minutes"]:
                    start = perf_counter()
                    lp_model = model(recovery, wait, downtime)
                    elapsed = perf_counter() - start
                    assembly.append(dict(id=recovery["id"], wait=wait, lag=downtime, rep=repeat, seconds=elapsed,
                                         variables=len(lp_model[3]), rows=len(lp_model[2])))
    write_csv(output / "model_construction.csv", assembly)
    by_model = defaultdict(list)
    for row in assembly:
        by_model[row["id"], row["wait"], row["lag"]].append(row["seconds"])
    profile_s = [sum(median(by_model[r["id"], w, d]) for w in P["wait_minutes"] for d in P["downtime_minutes"]) for r in profiles]
    (output / "model_cost.json").write_text(json.dumps(dict(total_s=sum(profile_s), profile_s=profile_s), indent=2) + "\n")
    print(f"{near_boundary} boundary queries, {mismatches} disagreements, {maximum_violation:.3g} maximum repair violation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Previously generated confirmation directory")
    audit(prepare_output(parser.parse_args().output))
