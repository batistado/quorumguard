#!/usr/bin/env python3
"""
QuorumGuard Component Ablation Study Evaluator.
===============================================
Evaluates the orthogonal contribution of each QuorumGuard mechanism over the
logged live MPS per-agent telemetry (`pilot/results/quorumguard_pilot.json`):

  1. Baseline (`Fast_k_of_N`): No entropy gate, no genealogy deduplication, no boost.
  2. `+ Entropy Gate Only`: Token-entropy & margin gating, but treats correlated echoes
     as independent votes and uses no straggler priority boosting.
  3. `+ Genealogy Independence Only`: Deduplicates shared provenance roots (`N_eff`),
     but lacks semantic entropy/margin gating and straggler boosting.
  4. `QuorumGuard (w/o Dissent Boost)`: Both Entropy Gate + Genealogy Independence,
     but `boost_speedup = 1.0x` (waits passively on straggler queue tail).
  5. `QuorumGuard (Full)`: Entropy Gate + Genealogy Independence + 3.2x Priority-Budget
     Dissent Straggler Boosting.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def _weight(agent: Dict[str, Any], use_entropy: bool) -> float:
    if not use_entropy:
        return 1.0
    cert_prob = max(0.0, 1.0 - float(agent["normalized_entropy"]))
    margin_conf = math.tanh(max(0.0, float(agent["logit_margin"])) / 3.5)
    base_conf = 0.55 * cert_prob + 0.45 * margin_conf
    if agent["tool_verified"]:
        mult = 1.38
    elif agent["tier"] == "verifier_7b":
        mult = 1.18
    elif agent["tier"] == "cot_7b":
        mult = 1.12
    elif agent["tier"] == "short_cot_3b":
        mult = 0.85
    else:
        mult = 0.76
    return min(1.35, max(0.05, base_conf * mult))


def eval_variant_on_stage(
    agents: List[Dict[str, Any]],
    gold: str,
    use_entropy: bool,
    use_genealogy: bool,
    use_boost: bool,
    boost_speedup: float = 3.2,
) -> Tuple[bool, float, bool]:
    n_total = len(agents)
    dyn_times = {int(a["agent_id"]): float(a["completion_time_s"]) for a in agents}
    by_id = {int(a["agent_id"]): a for a in agents}
    finished: List[int] = []
    boosted = False

    def aggregate(obs_list: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, int]]:
        if use_genealogy:
            rc_max: Dict[Tuple[str, str], float] = {}
            for a in obs_list:
                w = _weight(a, use_entropy)
                k = (str(a["genealogy_root"]), str(a["choice"]))
                rc_max[k] = max(rc_max.get(k, 0.0), w)
            tot: Dict[str, float] = {}
            rcnt: Dict[str, int] = {}
            for (_, ch), wm in rc_max.items():
                tot[ch] = tot.get(ch, 0.0) + wm
                rcnt[ch] = rcnt.get(ch, 0) + 1
            return tot, rcnt
        else:
            tot = {}
            rcnt = {}
            for a in obs_list:
                ch = str(a["choice"])
                tot[ch] = tot.get(ch, 0.0) + _weight(a, use_entropy)
                rcnt[ch] = rcnt.get(ch, 0) + 1
            return tot, rcnt

    while len(finished) < n_total:
        rem = [aid for aid in dyn_times if aid not in finished]
        nxt = min(rem, key=lambda aid: dyn_times[aid])
        t_now = dyn_times[nxt]
        finished.append(nxt)

        arrived = [by_id[aid] for aid in finished]
        tot, rcnt = aggregate(arrived)
        ranked = sorted(tot.items(), key=lambda kv: kv[1], reverse=True)
        lead_ch, lead_w = ranked[0]
        run_w = ranked[1][1] if len(ranked) > 1 else 0.0
        lead_r = rcnt[lead_ch]

        supporters = [a for a in arrived if a["choice"] == lead_ch]
        mean_ent = float(np.mean([float(a["normalized_entropy"]) for a in supporters]))
        min_mar = float(min(float(a["logit_margin"]) for a in supporters))
        any_dissent = len(ranked) >= 2 and run_w >= 0.35
        deep_roots = len(
            {
                (a["genealogy_root"] if use_genealogy else a["agent_id"])
                for a in supporters
                if a["tool_verified"] or a["tier"] in ("verifier_7b", "cot_7b")
            }
        )

        if not use_entropy and not use_genealogy:
            # Classic Fast k-of-N
            if lead_r >= 2 and len(finished) >= 2:
                return (lead_ch == gold), t_now, boosted
            if len(finished) >= 3:
                return (lead_ch == gold), t_now, boosted
            continue

        ent_ok = (mean_ent <= 0.08 and min_mar >= 3.2) if use_entropy else True
        tau_req = 2.12 if use_entropy else 2.95

        clean_fast = (not any_dissent) and lead_r >= 3 and lead_w >= tau_req and ent_ok
        verified_close = (
            deep_roots >= 2
            or (deep_roots >= 1 and not any_dissent)
        ) and lead_r >= 2 and (lead_w - run_w) >= 0.72

        if clean_fast or verified_close:
            return (lead_ch == gold), t_now, boosted

        if use_boost and (not boosted) and len(finished) >= 3:
            boosted = True
            for aid, ac in by_id.items():
                if aid not in finished and ac["tier"] in ("verifier_7b", "cot_7b"):
                    fl = t_now + 0.45 * float(ac["mps_wall_s"])
                    cp = t_now + (dyn_times[aid] - t_now) / boost_speedup
                    dyn_times[aid] = max(fl, cp)

    tot, _ = aggregate([by_id[aid] for aid in finished])
    win_ch = max(tot.items(), key=lambda kv: kv[1])[0]
    return (win_ch == gold), max(dyn_times.values()), boosted


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    pilot_path = root / "pilot" / "results" / "quorumguard_pilot.json"
    out_path = root / "pilot" / "results" / "ablation_results.json"

    data = json.loads(pilot_path.read_text())
    tasks = data["tasks"]

    variants = [
        ("1_Fast_k_of_N", False, False, False),
        ("2_Entropy_Gate_Only", True, False, False),
        ("3_Genealogy_Only", False, True, False),
        ("4_QuorumGuard_No_Boost", True, True, False),
        ("5_QuorumGuard_Full", True, True, True),
    ]

    summary: Dict[str, Any] = {}
    for name, u_ent, u_gen, u_bst in variants:
        oks, lats, trap_oks, bsts = [], [], [], []
        for t in tasks:
            ok, lat, bst = eval_variant_on_stage(
                t["stage1_agents"], t["stage1_gold"], u_ent, u_gen, u_bst
            )
            oks.append(ok)
            lats.append(lat)
            bsts.append(bst)
            if t["regime"] == "boundary_trap":
                trap_oks.append(ok)
        arr = np.array(lats, dtype=np.float64)
        p95 = float(np.percentile(arr, 95))
        cvar95 = float(np.mean(arr[arr >= p95]))
        summary[name] = {
            "stage1_accuracy": round(float(np.mean(oks)), 4),
            "boundary_trap_stage1_accuracy": round(float(np.mean(trap_oks)), 4) if trap_oks else 0.0,
            "mean_stage1_latency_s": round(float(np.mean(arr)), 4),
            "p95_stage1_latency_s": round(p95, 4),
            "cvar95_stage1_latency_s": round(cvar95, 4),
            "boost_rate": round(float(np.mean(bsts)), 4),
        }
        print(
            f"{name:24s}: S1_Acc={summary[name]['stage1_accuracy']*100:5.1f}% "
            f"(Trap={summary[name]['boundary_trap_stage1_accuracy']*100:5.1f}%) | "
            f"Mean={summary[name]['mean_stage1_latency_s']:5.2f}s | "
            f"P95={summary[name]['p95_stage1_latency_s']:5.2f}s | "
            f"CVaR95={summary[name]['cvar95_stage1_latency_s']:5.2f}s"
        )

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved ablations to {out_path}")


if __name__ == "__main__":
    main()
