from __future__ import annotations

import math
from statistics import mean, stdev
from typing import List

from .simulation import ComparisonSandbox, SIM_STEP_MS, normalize_site_count


def _summary(values: List[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mu = mean(values)
    sigma = stdev(values) if n > 1 else 0.0
    half = 1.96 * sigma / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean": round(mu, 6),
        "std": round(sigma, 6),
        "ci95_low": round(mu - half, 6),
        "ci95_high": round(mu + half, 6),
    }


def _significance(diff_summary: dict) -> str:
    low = diff_summary["ci95_low"]
    high = diff_summary["ci95_high"]
    if low > 0:
        return "astar_higher"
    if high < 0:
        return "baseline_higher"
    return "inconclusive"


def _build_conclusion(comp_rate_diff: dict, wait_diff: dict, distance_diff: dict) -> str:
    comp_sig = _significance(comp_rate_diff)
    wait_sig = _significance(wait_diff)
    dist_sig = _significance(distance_diff)

    lines = []

    if comp_sig == "astar_higher":
        lines.append("在当前轻量仿真设定下，A*+贪心的任务完成率在统计上高于A*基线。")
    elif comp_sig == "baseline_higher":
        lines.append("在当前轻量仿真设定下，A*基线的任务完成率在统计上高于A*+贪心。")
    else:
        lines.append("两种算法的任务完成率95%置信区间重叠，当前证据不足以下结论。")

    if wait_sig == "astar_higher":
        lines.append("平均等待时长差值（A*+贪心 - A*基线）显著为正，说明A*基线等待时长更低。")
    elif wait_sig == "baseline_higher":
        lines.append("平均等待时长差值（A*+贪心 - A*基线）显著为负，说明A*+贪心等待时长更低。")
    else:
        lines.append("平均等待时长差异不显著。")

    if dist_sig == "astar_higher":
        lines.append("总路程差值（A*+贪心 - A*基线）显著为正，说明A*基线路径成本更低。")
    elif dist_sig == "baseline_higher":
        lines.append("总路程差值（A*+贪心 - A*基线）显著为负，说明A*+贪心路径成本更低。")
    else:
        lines.append("总路程差异不显著。")

    lines.append("结论只对当前任务流、障碍布局与参数配置有效，外推到真实系统需高保真仿真或实机验证。")
    return " ".join(lines)


def run_evaluation(
    episodes: int = 24,
    horizon_s: float = 120.0,
    interval_s: float = 4.0,
    base_seed: int = 100,
    payload_capacity: int = 2,
    site_count: int = 3,
) -> dict:
    episodes = max(4, min(int(episodes), 200))
    horizon_ms = max(10000, int(float(horizon_s) * 1000))
    interval_ms = max(1000, int(float(interval_s) * 1000))
    payload_capacity = max(1, min(int(payload_capacity), 12))
    site_count = normalize_site_count(site_count)

    records = []
    checks = {
        "generated_equal": True,
        "completed_not_exceed_generated": True,
        "metrics_non_negative": True,
    }

    for i in range(episodes):
        sandbox = ComparisonSandbox(
            seed=base_seed + i,
            interval_ms=interval_ms,
            speed=1.0,
            payload_capacity=payload_capacity,
            site_count=site_count,
        )
        while sandbox.global_time_ms < horizon_ms:
            sandbox.advance_by(SIM_STEP_MS)

        snap = sandbox.metrics_snapshot()
        astar = snap["astar"]
        baseline = snap["mappo"]

        checks["generated_equal"] &= astar["generated"] == baseline["generated"]
        checks["completed_not_exceed_generated"] &= (
            astar["completed"] <= astar["generated"] and baseline["completed"] <= baseline["generated"]
        )

        for metric in [
            "avg_completion_s",
            "avg_wait_s",
            "completion_rate",
            "total_distance_px",
            "generated",
            "completed",
            "pending",
        ]:
            checks["metrics_non_negative"] &= astar[metric] >= 0 and baseline[metric] >= 0

        records.append({"seed": base_seed + i, "astar": astar, "baseline": baseline})

    astar_completion_rate = [r["astar"]["completion_rate"] for r in records]
    baseline_completion_rate = [r["baseline"]["completion_rate"] for r in records]
    astar_wait = [r["astar"]["avg_wait_s"] for r in records]
    baseline_wait = [r["baseline"]["avg_wait_s"] for r in records]
    astar_dist = [r["astar"]["total_distance_px"] for r in records]
    baseline_dist = [r["baseline"]["total_distance_px"] for r in records]

    diff_completion = [a - b for a, b in zip(astar_completion_rate, baseline_completion_rate)]
    diff_wait = [a - b for a, b in zip(astar_wait, baseline_wait)]
    diff_dist = [a - b for a, b in zip(astar_dist, baseline_dist)]

    summary = {
        "astar": {
            "completion_rate": _summary(astar_completion_rate),
            "avg_wait_s": _summary(astar_wait),
            "total_distance_px": _summary(astar_dist),
        },
        "baseline": {
            "completion_rate": _summary(baseline_completion_rate),
            "avg_wait_s": _summary(baseline_wait),
            "total_distance_px": _summary(baseline_dist),
        },
        "paired_diff_astar_minus_baseline": {
            "completion_rate": _summary(diff_completion),
            "avg_wait_s": _summary(diff_wait),
            "total_distance_px": _summary(diff_dist),
        },
    }

    conclusion = _build_conclusion(
        summary["paired_diff_astar_minus_baseline"]["completion_rate"],
        summary["paired_diff_astar_minus_baseline"]["avg_wait_s"],
        summary["paired_diff_astar_minus_baseline"]["total_distance_px"],
    )

    return {
        "episodes": episodes,
        "horizon_s": horizon_ms / 1000.0,
        "interval_s": interval_ms / 1000.0,
        "payload_capacity": payload_capacity,
        "site_count": int(site_count),
        "checks": checks,
        "summary": summary,
        "conclusion": conclusion,
        "samples": records[:6],
    }


def _winner_by_completion(report: dict) -> str:
    diff = report["summary"]["paired_diff_astar_minus_baseline"]["completion_rate"]
    sig = _significance(diff)
    if sig == "astar_higher":
        return "astar"
    if sig == "baseline_higher":
        return "baseline"
    return "inconclusive"


def run_payload_sweep(
    payload_values: List[int],
    episodes: int = 24,
    horizon_s: float = 120.0,
    interval_s: float = 4.0,
    base_seed: int = 100,
    site_count: int = 3,
) -> dict:
    site_count = normalize_site_count(site_count)
    capacities = sorted({max(1, min(int(v), 12)) for v in payload_values})
    if not capacities:
        capacities = [1, 2, 3, 4]

    reports = []
    for cap in capacities:
        rep = run_evaluation(
            episodes=episodes,
            horizon_s=horizon_s,
            interval_s=interval_s,
            base_seed=base_seed,
            payload_capacity=cap,
            site_count=site_count,
        )

        winner = _winner_by_completion(rep)
        reports.append(
            {
                "payload_capacity": cap,
                "winner_by_completion": winner,
                "astar_completion_mean": rep["summary"]["astar"]["completion_rate"]["mean"],
                "baseline_completion_mean": rep["summary"]["baseline"]["completion_rate"]["mean"],
                "astar_wait_mean": rep["summary"]["astar"]["avg_wait_s"]["mean"],
                "baseline_wait_mean": rep["summary"]["baseline"]["avg_wait_s"]["mean"],
                "astar_distance_mean": rep["summary"]["astar"]["total_distance_px"]["mean"],
                "baseline_distance_mean": rep["summary"]["baseline"]["total_distance_px"]["mean"],
                "completion_diff_ci95": [
                    rep["summary"]["paired_diff_astar_minus_baseline"]["completion_rate"]["ci95_low"],
                    rep["summary"]["paired_diff_astar_minus_baseline"]["completion_rate"]["ci95_high"],
                ],
                "raw_report": rep,
            }
        )

    astar_better = sum(1 for r in reports if r["winner_by_completion"] == "astar")
    baseline_better = sum(1 for r in reports if r["winner_by_completion"] == "baseline")
    inconclusive = sum(1 for r in reports if r["winner_by_completion"] == "inconclusive")

    if astar_better > baseline_better:
        overall = "在扫描的载荷上限范围内，A*+贪心在更多载荷档位上完成率更优。"
    elif baseline_better > astar_better:
        overall = "在扫描的载荷上限范围内，A*基线在更多载荷档位上完成率更优。"
    else:
        overall = "在扫描的载荷上限范围内，两算法在完成率上的优势档位数量接近。"

    overall += " 结果应结合等待时长与总路程共同解释，不建议只看单一指标。"

    return {
        "mode": "payload_sweep",
        "episodes": max(4, min(int(episodes), 200)),
        "horizon_s": max(10.0, float(horizon_s)),
        "interval_s": max(1.0, float(interval_s)),
        "site_count": int(site_count),
        "payload_values": capacities,
        "aggregate": {
            "astar_better_count": astar_better,
            "baseline_better_count": baseline_better,
            "inconclusive_count": inconclusive,
        },
        "overall_conclusion": overall,
        "capacity_reports": reports,
    }
