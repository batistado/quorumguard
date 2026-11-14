#!/usr/bin/env python3
"""
Live Concurrent KV-Cache-Resident Quantum-Preemptive MPS Server Benchmark
=========================================================================
Runs an in-process live multi-request GPU inference server on Apple M5 Pro
(`Qwen/Qwen2.5-7B-Instruct`, FP16) where requests arrive in real wall-clock
time according to a Poisson process (target utilization rho = 0.82, N = 64
requests/policy: 68% foreground normal turns of 24-40 tokens and 32% deep
verifier stragglers of 96-136 tokens, half triggering dissent boost).

Unlike trace-only queueing simulations, every request runs real FP16 Metal
prefill and 8-token decode quanta (`torch.mps.synchronize()`), with preempted
requests holding live `DynamicCache` KV states simultaneously in unified
memory (`torch.mps.current_allocated_memory()` tracked continuously) until
resumed without KV recomputation.

Outputs live hardware server metrics to
`benchmarks/results/live_mps_quantum_server.json`.
"""

from __future__ import annotations

import gc
import heapq
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from quorumguard.scheduler import PriorityBudgetPool  # noqa: E402


PROMPTS = [
    "Verify whether uint16_t x = 256 * 256 overflows in C and state the exact wrapped value.",
    "Check SQL query behavior when NOT IN subquery contains a NULL element.",
    "Analyze IEEE-754 float32 precision loss when adding 1.0 to 16777216.0.",
    "Evaluate Python late-binding lambda closure list comprehension behavior.",
    "Check signed 32-bit integer overflow in binary search midpoint (low + high) / 2.",
    "Determine Shortest Path under Dijkstra with negative-weight edges.",
]


@dataclass
class LiveRequestSpec:
    req_id: int
    arrival_offset_s: float
    is_straggler: bool
    dissent_triggered: bool
    target_quanta: int
    prompt_idx: int


def calibrate_service_baseline(
    model: AutoModelForCausalLM,
    prompt_tensors: List[torch.Tensor],
    quantum_tokens: int = 8,
) -> Tuple[float, float]:
    """Warm up MPS kernels and measure baseline prefill and 8-token quantum durations."""
    pref_times: List[float] = []
    quant_times: List[float] = []
    with torch.inference_mode():
        for i in range(6):
            inp = prompt_tensors[i % len(prompt_tensors)]
            torch.mps.synchronize()
            t0 = time.perf_counter()
            out = model(input_ids=inp, use_cache=True)
            torch.mps.synchronize()
            pref_times.append(time.perf_counter() - t0)
            past = out.past_key_values
            nxt = torch.argmax(out.logits[:, -1:, :], dim=-1)
            for _ in range(3):
                torch.mps.synchronize()
                tq0 = time.perf_counter()
                for _ in range(quantum_tokens):
                    out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    nxt = torch.argmax(out.logits[:, -1:, :], dim=-1)
                torch.mps.synchronize()
                quant_times.append(time.perf_counter() - tq0)
            del past, out, nxt
    torch.mps.empty_cache()
    # Skip first cold iteration
    return float(np.mean(pref_times[1:])), float(np.mean(quant_times[3:]))


def run_live_policy_trace(
    model: AutoModelForCausalLM,
    prompt_tensors: List[torch.Tensor],
    specs: List[LiveRequestSpec],
    policy: str,
    quantum_tokens: int = 8,
) -> Dict[str, Any]:
    """
    Executes live wall-clock Poisson arrival stream against resident KV caches on MPS.
    """
    n_req = len(specs)
    budget_pool = PriorityBudgetPool(max_budget_tokens=6.0, refill_per_stage=0.32)
    rng_u = np.random.default_rng(99)

    prio_map: Dict[int, int] = {}
    for sp in specs:
        if policy == "FCFS_Default":
            prio_map[sp.req_id] = 0
        elif policy == "Unbounded_Urgent":
            prio_map[sp.req_id] = -10 if rng_u.random() < 0.65 else 0
        elif policy == "QuorumGuard_Budgeted":
            if sp.is_straggler and sp.dissent_triggered and budget_pool.try_acquire(1) > 0:
                prio_map[sp.req_id] = -10
            else:
                prio_map[sp.req_id] = 0
            budget_pool.on_stage_complete()
        else:
            raise ValueError(f"Unknown policy: {policy}")

    spec_by_id = {sp.req_id: sp for sp in specs}
    pending = sorted(specs, key=lambda s: s.arrival_offset_s)
    cursor = 0

    ready_heap: List[Tuple[int, float, int]] = []
    kv_caches: Dict[int, Any] = {}
    next_tokens: Dict[int, torch.Tensor] = {}
    quanta_done: Dict[int, int] = {sp.req_id: 0 for sp in specs}
    exec_service_s: Dict[int, float] = {sp.req_id: 0.0 for sp in specs}
    completion_s: Dict[int, float] = {}

    max_concurrent_kv = 0
    peak_alloc_bytes = torch.mps.current_allocated_memory()
    peak_driver_bytes = torch.mps.driver_allocated_memory()
    preemptions_count = 0
    last_served_rid = -1

    gc.collect()
    torch.mps.empty_cache()
    t_epoch = time.perf_counter()

    with torch.inference_mode():
        while len(completion_s) < n_req:
            t_now = time.perf_counter() - t_epoch
            while cursor < n_req and pending[cursor].arrival_offset_s <= t_now:
                sp = pending[cursor]
                heapq.heappush(ready_heap, (prio_map[sp.req_id], sp.arrival_offset_s, sp.req_id))
                cursor += 1

            if not ready_heap:
                wait_gap = pending[cursor].arrival_offset_s - (time.perf_counter() - t_epoch)
                if wait_gap > 0:
                    time.sleep(wait_gap)
                continue

            prio, arr_off, rid = heapq.heappop(ready_heap)
            sp = spec_by_id[rid]
            if last_served_rid not in (-1, rid) and last_served_rid in kv_caches:
                preemptions_count += 1
            last_served_rid = rid

            # Execute prefill if first touch
            if rid not in kv_caches:
                inp = prompt_tensors[sp.prompt_idx]
                torch.mps.synchronize()
                ts = time.perf_counter()
                out = model(input_ids=inp, use_cache=True)
                torch.mps.synchronize()
                exec_service_s[rid] += time.perf_counter() - ts
                kv_caches[rid] = out.past_key_values
                next_tokens[rid] = torch.argmax(out.logits[:, -1:, :], dim=-1)
                max_concurrent_kv = max(max_concurrent_kv, len(kv_caches))

            past = kv_caches[rid]
            nxt = next_tokens[rid]

            if policy == "FCFS_Default":
                # Non-preemptive run-to-completion
                torch.mps.synchronize()
                ts = time.perf_counter()
                while quanta_done[rid] < sp.target_quanta:
                    for _ in range(quantum_tokens):
                        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                        past = out.past_key_values
                        nxt = torch.argmax(out.logits[:, -1:, :], dim=-1)
                    quanta_done[rid] += 1
                torch.mps.synchronize()
                exec_service_s[rid] += time.perf_counter() - ts
            else:
                # Single 8-token quantum before yielding for higher-priority arrivals
                torch.mps.synchronize()
                ts = time.perf_counter()
                for _ in range(quantum_tokens):
                    out = model(input_ids=nxt, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    nxt = torch.argmax(out.logits[:, -1:, :], dim=-1)
                torch.mps.synchronize()
                exec_service_s[rid] += time.perf_counter() - ts
                quanta_done[rid] += 1
                kv_caches[rid] = past
                next_tokens[rid] = nxt

            peak_alloc_bytes = max(peak_alloc_bytes, torch.mps.current_allocated_memory())
            peak_driver_bytes = max(peak_driver_bytes, torch.mps.driver_allocated_memory())

            if quanta_done[rid] >= sp.target_quanta:
                completion_s[rid] = time.perf_counter() - t_epoch
                kv_caches.pop(rid, None)
                next_tokens.pop(rid, None)
                if len(completion_s) % 16 == 0 or len(completion_s) == n_req:
                    print(
                        f"    [{policy}] {len(completion_s)}/{n_req} done | "
                        f"live_KV={len(kv_caches)} (max={max_concurrent_kv}), "
                        f"MPS_alloc={peak_alloc_bytes/1e9:.2f}GB",
                        flush=True,
                    )
            else:
                heapq.heappush(ready_heap, (prio, arr_off, rid))

    gc.collect()
    torch.mps.empty_cache()

    dissent_turn = np.array(
        [completion_s[s.req_id] - s.arrival_offset_s for s in specs if s.is_straggler and s.dissent_triggered]
    )
    dissent_wait = np.array(
        [
            max(0.0, (completion_s[s.req_id] - s.arrival_offset_s) - exec_service_s[s.req_id])
            for s in specs
            if s.is_straggler and s.dissent_triggered
        ]
    )
    normal_turn = np.array(
        [completion_s[s.req_id] - s.arrival_offset_s for s in specs if not s.is_straggler]
    )

    def _cvar95(x: np.ndarray) -> float:
        p95 = np.percentile(x, 95)
        return float(np.mean(x[x >= p95]))

    return {
        "num_requests": n_req,
        "preemptions": preemptions_count,
        "max_concurrent_kv_caches": max_concurrent_kv,
        "peak_mps_allocated_gb": round(peak_alloc_bytes / 1e9, 3),
        "peak_mps_driver_gb": round(peak_driver_bytes / 1e9, 3),
        "dissent_straggler_mean_turnaround_s": round(float(np.mean(dissent_turn)), 4),
        "dissent_straggler_p95_turnaround_s": round(float(np.percentile(dissent_turn, 95)), 4),
        "dissent_straggler_cvar95_turnaround_s": round(_cvar95(dissent_turn), 4),
        "dissent_straggler_mean_queue_wait_s": round(float(np.mean(dissent_wait)), 4),
        "foreground_normal_mean_turnaround_s": round(float(np.mean(normal_turn)), 4),
        "foreground_normal_p95_turnaround_s": round(float(np.percentile(normal_turn, 95)), 4),
        "total_wall_clock_s": round(max(completion_s.values()), 3),
    }


def main() -> None:
    out_path = ROOT / "benchmarks" / "results" / "live_mps_quantum_server.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    quantum_tokens = 8
    n_requests = 64
    target_rho = 0.82

    print(f"[LiveServer] Loading {model_name} on Apple M5 Pro MPS (FP16)...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, local_files_only=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, local_files_only=True)
    model.to("mps")
    model.eval()
    torch.mps.synchronize()

    prompt_tensors: List[torch.Tensor] = []
    for p_txt in PROMPTS:
        enc = tok.apply_chat_template(
            [{"role": "system", "content": "Reason concisely."}, {"role": "user", "content": p_txt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        ids = list(enc["input_ids"]) if hasattr(enc, "keys") and "input_ids" in enc else list(enc)
        prompt_tensors.append(torch.tensor([ids], dtype=torch.long, device="mps"))

    mean_pref_s, mean_q_s = calibrate_service_baseline(model, prompt_tensors, quantum_tokens=quantum_tokens)
    print(
        f"[LiveServer] Calibrated MPS baseline: prefill={mean_pref_s*1000:.1f}ms, "
        f"8-tok quantum={mean_q_s*1000:.1f}ms",
        flush=True,
    )

    rng = np.random.default_rng(20261114)
    specs: List[LiveRequestSpec] = []
    est_services: List[float] = []
    raw_flags: List[Tuple[bool, bool, int, int]] = []
    for i in range(n_requests):
        is_strag = bool(rng.random() < 0.32)
        dissent = bool(is_strag and (rng.random() < 0.50))
        nq = int(rng.integers(12, 18)) if is_strag else int(rng.integers(3, 6))
        p_idx = int(rng.integers(0, len(prompt_tensors)))
        est_services.append(mean_pref_s + nq * mean_q_s)
        raw_flags.append((is_strag, dissent, nq, p_idx))

    mean_srv = float(np.mean(est_services))
    lam = target_rho / mean_srv
    arr_offsets = np.cumsum(rng.exponential(scale=1.0 / lam, size=n_requests))
    # Shift start so first arrival enters immediately at t=0
    arr_offsets = arr_offsets - arr_offsets[0]
    for i, (is_strag, dissent, nq, p_idx) in enumerate(raw_flags):
        specs.append(
            LiveRequestSpec(
                req_id=i,
                arrival_offset_s=float(arr_offsets[i]),
                is_straggler=is_strag,
                dissent_triggered=dissent,
                target_quanta=nq,
                prompt_idx=p_idx,
            )
        )

    print(
        f"[LiveServer] Running N={n_requests} Poisson requests (rho={target_rho:.2f}, "
        f"est_mean_srv={mean_srv:.2f}s, lam={lam:.3f} req/s) across 3 policies...",
        flush=True,
    )
    pol_results: Dict[str, Any] = {}
    for pol in ("FCFS_Default", "Unbounded_Urgent", "QuorumGuard_Budgeted"):
        print(f"  -> Executing live KV-cache policy: {pol}", flush=True)
        pol_results[pol] = run_live_policy_trace(
            model=model,
            prompt_tensors=prompt_tensors,
            specs=specs,
            policy=pol,
            quantum_tokens=quantum_tokens,
        )

    fcfs = pol_results["FCFS_Default"]
    unb = pol_results["Unbounded_Urgent"]
    qg = pol_results["QuorumGuard_Budgeted"]

    payload = {
        "hardware": "Apple M5 Pro MPS (48 GB Unified Memory, FP16)",
        "model": model_name,
        "quantum_tokens": quantum_tokens,
        "num_requests_per_policy": n_requests,
        "target_utilization_rho": target_rho,
        "calibrated_prefill_ms": round(mean_pref_s * 1000.0, 2),
        "calibrated_8tok_quantum_ms": round(mean_q_s * 1000.0, 2),
        "policies": pol_results,
        "summary_comparisons": {
            "queue_wait_compression_x": round(
                fcfs["dissent_straggler_mean_queue_wait_s"]
                / max(1e-4, qg["dissent_straggler_mean_queue_wait_s"]),
                3,
            ),
            "p95_turnaround_speedup_x": round(
                fcfs["dissent_straggler_p95_turnaround_s"]
                / max(1e-4, qg["dissent_straggler_p95_turnaround_s"]),
                3,
            ),
            "quorumguard_foreground_p95_impact_pct": round(
                (qg["foreground_normal_p95_turnaround_s"] - fcfs["foreground_normal_p95_turnaround_s"])
                / max(1e-4, fcfs["foreground_normal_p95_turnaround_s"])
                * 100.0,
                2,
            ),
            "unbounded_foreground_p95_impact_pct": round(
                (unb["foreground_normal_p95_turnaround_s"] - fcfs["foreground_normal_p95_turnaround_s"])
                / max(1e-4, fcfs["foreground_normal_p95_turnaround_s"])
                * 100.0,
                2,
            ),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[LiveServer] Saved {out_path}", flush=True)
    print(json.dumps(payload["summary_comparisons"], indent=2), flush=True)


if __name__ == "__main__":
    main()
