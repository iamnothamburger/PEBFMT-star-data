#!/usr/bin/env python3
"""Recompute and verify manuscript statistics from the public run-level data."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import math
from pathlib import Path
import statistics
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
Z_95 = 1.959963984540054


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def optional_float(raw: object) -> Optional[float]:
    value = str(raw).strip()
    if value == "" or value.lower() in {"nan", "inf", "+inf", "-inf"}:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def as_bool(raw: object) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes"}


def wilson_interval(successes: int, total: int) -> Tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive denominator")
    proportion = successes / total
    denominator = 1.0 + Z_95 * Z_95 / total
    center = (proportion + Z_95 * Z_95 / (2.0 * total)) / denominator
    half = (
        Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + Z_95 * Z_95 / (4.0 * total * total)
        )
        / denominator
    )
    return center - half, center + half


def assert_close(
    actual: Optional[float],
    expected: Optional[float],
    label: str,
    tolerance: float = 1e-9,
) -> None:
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise AssertionError(f"{label}: expected {expected!r}, found {actual!r}")
        return
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{label}: expected {expected:.15g}, found {actual:.15g}")


def assert_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, found {actual!r}")


def sample_sd(values: Sequence[float]) -> Optional[float]:
    return statistics.stdev(values) if len(values) >= 2 else None


def verify_table3(main_rows: Sequence[Dict[str, str]]) -> None:
    stored_rows = read_csv(PROJECT_ROOT / "results" / "table3_statistics.csv")
    stored = {(row["scene"], row["planner"]): row for row in stored_rows}
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in main_rows:
        grouped[(row["scene"], row["planner_public"])].append(row)
    assert_equal(len(grouped), 20, "Table 3 group count")
    assert_equal(len(stored), 20, "stored Table 3 row count")

    for key, rows in grouped.items():
        if key not in stored:
            raise AssertionError(f"missing stored Table 3 row: {key}")
        expected = stored[key]
        runs = len(rows)
        solved = [row for row in rows if as_bool(row["solved"])]
        hits = [row for row in rows if as_bool(row["target_hit"])]
        hit_times = [
            value
            for value in (optional_float(row["target_time_s"]) for row in hits)
            if value is not None
        ]
        final_costs = [
            value
            for value in (optional_float(row["final_cost"]) for row in solved)
            if value is not None
        ]
        solved_ci = wilson_interval(len(solved), runs)
        hit_ci = wilson_interval(len(hits), runs)

        label = f"Table 3 {key[0]}/{key[1]}"
        assert_equal(runs, int(expected["runs"]), f"{label} runs")
        assert_equal(len(solved), int(expected["solved_n"]), f"{label} solved_n")
        assert_equal(len(hits), int(expected["hit_n"]), f"{label} hit_n")
        assert_equal(len(hit_times), int(expected["target_time_n"]), f"{label} target_time_n")
        assert_equal(len(final_costs), int(expected["final_cost_n"]), f"{label} final_cost_n")
        assert_close(
            statistics.fmean(hit_times) if hit_times else None,
            optional_float(expected["target_time_mean"]),
            f"{label} target_time_mean",
        )
        assert_close(
            sample_sd(hit_times),
            optional_float(expected["target_time_sd"]),
            f"{label} target_time_sd",
        )
        assert_close(
            statistics.median(hit_times) if hit_times else None,
            optional_float(expected["target_time_median"]),
            f"{label} target_time_median",
        )
        assert_close(
            statistics.fmean(final_costs) if final_costs else None,
            optional_float(expected["final_cost_mean"]),
            f"{label} final_cost_mean",
        )
        assert_close(
            sample_sd(final_costs),
            optional_float(expected["final_cost_sd"]),
            f"{label} final_cost_sd",
        )
        assert_close(len(solved) / runs, float(expected["solved_rate"]), f"{label} solved_rate")
        assert_close(solved_ci[0], float(expected["solved_ci_low"]), f"{label} solved_ci_low")
        assert_close(solved_ci[1], float(expected["solved_ci_high"]), f"{label} solved_ci_high")
        assert_close(len(hits) / runs, float(expected["hit_rate"]), f"{label} hit_rate")
        assert_close(hit_ci[0], float(expected["hit_ci_low"]), f"{label} hit_ci_low")
        assert_close(hit_ci[1], float(expected["hit_ci_high"]), f"{label} hit_ci_high")


def verify_table4(ablation_rows: Sequence[Dict[str, str]]) -> None:
    stored_rows = read_csv(PROJECT_ROOT / "results" / "table4_statistics.csv")
    stored = {row["planner"]: row for row in stored_rows}
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in ablation_rows:
        grouped[row["planner_public"]].append(row)
    assert_equal(set(grouped), set(stored), "Table 4 planner set")

    expected_hits = {
        "BFMT*": 78,
        "PEBFMT*-global-only": 231,
        "PEBFMT*-piecewise-only": 229,
        "PEBFMT*": 231,
    }
    for planner, rows in grouped.items():
        expected = stored[planner]
        hits = [row for row in rows if as_bool(row["target_hit"])]
        target_times = [
            value
            for value in (optional_float(row["target_time_s"]) for row in hits)
            if value is not None
        ]
        normalized_costs = [
            value
            for value in (
                optional_float(row["final_cost_normalized"]) for row in rows
            )
            if value is not None
        ]
        hit_ci = wilson_interval(len(hits), len(rows))
        label = f"Table 4 {planner}"
        assert_equal(len(rows), 250, f"{label} public run count")
        assert_equal(len(rows), int(expected["runs"]), f"{label} stored run count")
        assert_equal(len(hits), expected_hits[planner], f"{label} manuscript hit count")
        assert_equal(len(hits), int(expected["target_hits"]), f"{label} stored hit count")
        assert_close(
            len(hits) / len(rows),
            float(expected["target_hit_rate"]),
            f"{label} target_hit_rate",
        )
        assert_close(
            hit_ci[0], float(expected["target_hit_wilson_low"]), f"{label} Wilson low"
        )
        assert_close(
            hit_ci[1], float(expected["target_hit_wilson_high"]), f"{label} Wilson high"
        )
        assert_close(
            statistics.fmean(normalized_costs),
            float(expected["mean_normalized_final_cost"]),
            f"{label} normalized final cost",
        )
        assert_close(
            statistics.median(target_times),
            float(expected["conditional_median_runtime_to_threshold"]),
            f"{label} conditional median target time",
        )
        for field in (
            "mean_normalized_final_cost_bootstrap_low",
            "mean_normalized_final_cost_bootstrap_high",
            "conditional_median_runtime_bootstrap_low",
            "conditional_median_runtime_bootstrap_high",
        ):
            if optional_float(expected[field]) is None:
                raise AssertionError(f"{label}: locked manuscript interval {field} is missing")


def count_trace_rows(path: Path) -> Tuple[int, Dict[Tuple[str, str, str, str], int]]:
    counts: Dict[Tuple[str, str, str, str], int] = Counter()
    rows = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            key = (
                row["scene_raw"],
                row["planner_raw"],
                row["run_id"],
                row["source_log_id"],
            )
            counts[key] += 1
    return rows, counts


def verify_robot_if_present() -> None:
    trials_path = PROJECT_ROOT / "data" / "robot" / "trial_results.csv"
    summary_path = PROJECT_ROOT / "results" / "robot_summary.csv"
    if not trials_path.exists() or not summary_path.exists():
        raise AssertionError("robot trial data and summary must both be present")
    rows = read_csv(trials_path)
    assert_equal(len(rows), 120, "robot trial count")
    assert_equal(
        len({(row["planner_id"], row["trial_id"]) for row in rows}),
        120,
        "robot planner/trial key count",
    )
    if any(row[column] != "" for row in rows for column in ("obj_x", "obj_y", "obj_z", "obj_yaw")):
        raise AssertionError("robot object-coordinate columns must remain empty")

    planner_map = {
        "DAFMTkConfigDefault": ("DAFMT*", "PEBFMT*"),
        "BFMTkConfigDefault": ("BFMT*", "BFMT*"),
        "BITstarkConfigDefault": ("BIT*", "BIT*"),
        "InformedRRTstarkConfigDefault": ("Informed-RRT*", "Informed RRT*"),
    }
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["planner_id"] not in planner_map:
            raise AssertionError(f"unknown robot planner identifier: {row['planner_id']}")
        raw_label, _ = planner_map[row["planner_id"]]
        assert_equal(row["planner_label"], raw_label, "robot planner label")
        grouped[row["planner_id"]].append(row)

        pick = as_bool(row["pick_success"])
        place = as_bool(row["place_success"])
        reset = as_bool(row["reset_success"])
        if place and not pick:
            raise AssertionError("robot Place success cannot precede Pick success")
        if reset and not place:
            raise AssertionError("robot Reset success cannot precede Place success")
        if reset:
            for total, parts in (
                ("total_plan_time", ("pick_plan_time", "place_plan_time", "reset_plan_time")),
                ("total_path_cost", ("pick_path_cost", "place_path_cost", "reset_path_cost")),
                ("total_waypoints", ("pick_waypoints", "place_waypoints", "reset_waypoints")),
            ):
                assert_close(
                    float(row[total]),
                    sum(float(row[field]) for field in parts),
                    f"robot {row['planner_id']} trial {row['trial_id']} {total}",
                    tolerance=1e-6,
                )

    assert_equal(set(grouped), set(planner_map), "robot planner set")
    for planner_id, planner_rows in grouped.items():
        assert_equal(len(planner_rows), 30, f"robot {planner_id} trial count")

    pebfmt = grouped["DAFMTkConfigDefault"]
    full = [
        row
        for row in pebfmt
        if as_bool(row["pick_success"])
        and as_bool(row["place_success"])
        and as_bool(row["reset_success"])
    ]
    costs = [float(row["total_path_cost"]) for row in full]
    waypoints = [float(row["total_waypoints"]) for row in full]
    assert_equal(len(full), 26, "PEBFMT robot full successes")
    assert_close(statistics.median(costs), 17.8507755, "PEBFMT robot median path cost")
    assert_close(statistics.median(waypoints), 97.0, "PEBFMT robot median waypoints")

    summary_rows = {row["planner"]: row for row in read_csv(summary_path)}
    assert_equal(len(summary_rows), 4, "robot summary row count")
    for planner_id, planner_rows in grouped.items():
        _, public_label = planner_map[planner_id]
        expected = summary_rows[public_label]
        full_rows = [row for row in planner_rows if as_bool(row["reset_success"])]

        def mean_for(field: str, selected: Sequence[Dict[str, str]]) -> Optional[float]:
            values = [float(row[field]) for row in selected if row[field] != ""]
            return statistics.fmean(values) if values else None

        assert_equal(len(planner_rows), int(expected["trials"]), f"robot {public_label} trials")
        for stage in ("pick", "place", "reset"):
            rate = sum(as_bool(row[f"{stage}_success"]) for row in planner_rows) / len(planner_rows)
            assert_close(rate, float(expected[f"{stage}_success_rate"]), f"robot {public_label} {stage} rate")
            selected = [row for row in planner_rows if as_bool(row[f"{stage}_success"])]
            assert_close(
                mean_for(f"{stage}_plan_time", selected),
                optional_float(expected[f"mean_{stage}_plan_time"]),
                f"robot {public_label} mean {stage} time",
            )
        for source_field, summary_field in (
            ("total_plan_time", "mean_total_time"),
            ("total_path_cost", "mean_total_path_cost"),
            ("total_waypoints", "mean_total_waypoints"),
        ):
            assert_close(
                mean_for(source_field, full_rows),
                optional_float(expected[summary_field]),
                f"robot {public_label} {summary_field}",
            )
    print("Robot data: 120 trials and summary verified.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-trace-counts",
        action="store_true",
        help="Skip the compressed 1-second trace row-count check.",
    )
    args = parser.parse_args()

    main_rows = read_csv(PROJECT_ROOT / "data" / "main-study" / "main_runs.csv")
    ablation_rows = read_csv(PROJECT_ROOT / "data" / "ablation" / "ablation_runs.csv")
    assert_equal(len(main_rows), 1000, "main-study public run count")
    assert_equal(len(ablation_rows), 1000, "ablation public run count")
    assert_equal(
        len({(row["scene_raw"], row["planner_raw"], row["run_id"]) for row in main_rows}),
        1000,
        "main-study run-key count",
    )
    assert_equal(
        len(
            {
                (row["scene_raw"], row["planner_raw"], row["run_id"])
                for row in ablation_rows
            }
        ),
        1000,
        "ablation run-key count",
    )

    verify_table3(main_rows)
    verify_table4(ablation_rows)

    pebfmt = [row for row in main_rows if row["planner_public"] == "PEBFMT*"]
    pebfmt_hits = [row for row in pebfmt if as_bool(row["target_hit"])]
    pebfmt_times = [float(row["target_time_s"]) for row in pebfmt_hits]
    assert_equal(len(pebfmt), 250, "main PEBFMT run count")
    assert_equal(len(pebfmt_hits), 237, "main PEBFMT target-hit count")
    assert_close(
        statistics.fmean(pebfmt_times),
        14.940715,
        "main PEBFMT pooled conditional mean target time",
        tolerance=5e-7,
    )
    assert_close(
        statistics.median(pebfmt_times),
        7.770010,
        "main PEBFMT pooled conditional median target time",
        tolerance=5e-7,
    )

    if not args.skip_trace_counts:
        for label, path in (
            (
                "main",
                PROJECT_ROOT / "data" / "main-study" / "main_trace_1s.csv.gz",
            ),
            (
                "ablation",
                PROJECT_ROOT / "data" / "ablation" / "ablation_trace_1s.csv.gz",
            ),
        ):
            row_count, per_run = count_trace_rows(path)
            assert_equal(row_count, 61000, f"{label} trace row count")
            assert_equal(len(per_run), 1000, f"{label} trace run-key count")
            if set(per_run.values()) != {61}:
                raise AssertionError(f"{label}: every run must have exactly 61 trace rows")

    verify_robot_if_present()
    print("Main study: 1000 runs; PEBFMT*=237/250 target hits.")
    print(
        "Main PEBFMT* conditional target time: "
        f"mean={statistics.fmean(pebfmt_times):.6f} s, "
        f"median={statistics.median(pebfmt_times):.6f} s."
    )
    print("Ablation target hits: BFMT*=78, global-only=231, piecewise-only=229, full=231.")
    if not args.skip_trace_counts:
        print("Trace files: 61,000 rows each (61 one-second points for every run).")
    print("All public-data and manuscript-statistic checks passed.")


if __name__ == "__main__":
    main()
