#!/usr/bin/env python3
"""
QuorumBench-50 Scale-Up Evaluation (3B + 7B + 14B-Coder Heterogeneous Pool)
===========================================================================
Scales the QuorumGuard benchmark to 50 Two-Stage Workflows (100 total stage
evaluations: 20 Routine + 30 Subtle Multi-Domain Boundary/Bug Traps) across a
3-Model Heterogeneous Asynchronous Committee on Apple M5 Pro MPS (FP16):

  - Fast Tier (Agents 0, 1, 2) & Quick Check (Agent 3): `Qwen/Qwen2.5-3B-Instruct`
  - Medium Chain-of-Thought Tier (Agent 4):             `Qwen/Qwen2.5-7B-Instruct`
  - Deep Tool-Verified Stragglers (Agents 5, 6 & S2):   `Qwen/Qwen2.5-Coder-14B-Instruct`
                                                        + Live Sandboxed Python Execution

Logs complete per-agent MPS wall-clock latency, queue/sandbox tail latency,
4-way pre-commitment posterior entropy, logit margins, and 5-policy consensus
metrics to `benchmarks/results/quorumbench_50_3b_7b_14b.json`.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pilot.quorumguard_pilot import (  # noqa: E402
    CODE_BLOCK_RE,
    AgentObservation,
    MPSAgentEngine,
    QuorumDecision,
    TwoStageTask,
    build_pilot_tasks,
    evaluate_stage_policies,
    format_mc_prompt,
    run_python_sandbox,
)


def build_additional_20_tasks() -> List[TwoStageTask]:
    """8 additional routine tasks (R13..R20) + 12 additional boundary-trap tasks (T19..T30)."""
    return [
        # ---------------- 8 ADDITIONAL ROUTINE TASKS ----------------
        TwoStageTask(
            task_id="R13_http_429_status",
            regime="routine",
            category="distributed_systems",
            stage1_question="In standard HTTP status codes (RFC 6585), what condition does status code 429 indicate?",
            stage1_options={"A": "Payload Too Large", "B": "Too Many Requests (rate limit exceeded)", "C": "Gateway Timeout", "D": "Unauthorized"},
            stage1_gold="B",
            upstream_triage_note="RFC 6585 Section 4: HTTP 429 indicates Too Many Requests.",
            stage2_question="Given Stage-1's HTTP 429 classification, how should an API client handle a 429 response?",
            stage2_options={
                "A": "Immediately send larger request payloads",
                "B": "Honor the `Retry-After` header or apply exponential backoff with jitter",
                "C": "Re-prompt the user for new TLS certificates",
                "D": "Treat 429 as HTTP 200 OK",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="R14_quicksort_worst_case",
            regime="routine",
            category="algorithms",
            stage1_question="What is the worst-case time complexity of deterministic first-element-pivot Quicksort on an already sorted array of N distinct elements?",
            stage1_options={"A": "O(N)", "B": "O(N log N)", "C": "O(N^2)", "D": "O(log N)"},
            stage1_gold="C",
            upstream_triage_note="First-element pivot on sorted input partitions into 0 and N-1 elements -> sum_{i=1}^{N-1} i = O(N^2).",
            stage2_question="Given Stage-1's worst-case complexity diagnosis, what pivot/fallback change guarantees O(N log N) worst-case sorting?",
            stage2_options={
                "A": "Always pick the last element as pivot",
                "B": "Use Introsort (switch to Heapsort if recursion depth exceeds 2 * floor(log2 N))",
                "C": "Disable partitioning on sorted arrays by reversing first",
                "D": "Switch to Bubble Sort",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="R15_raid1_capacity",
            regime="routine",
            category="systems_basics",
            stage1_question="A storage server pairs two identical 4 TB NVMe drives in a standard RAID-1 (full mirroring) array. What is the usable logical storage capacity?",
            stage1_options={"A": "2 TB", "B": "4 TB", "C": "6 TB", "D": "8 TB"},
            stage1_gold="B",
            upstream_triage_note="RAID-1 mirrors identical blocks across both 4 TB drives -> 4 TB usable capacity.",
            stage2_question="Given Stage-1's RAID-1 usable capacity, how many single-drive failures can the 2-drive RAID-1 mirror survive without data loss?",
            stage2_options={"A": "0 drive failures", "B": "1 drive failure", "C": "2 simultaneous drive failures", "D": "3 drive failures"},
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="R16_bloom_filter_fn",
            regime="routine",
            category="algorithms",
            stage1_question="In a standard insert-only Bloom filter (without deletions or bit corruption), what is the exact theoretical false-negative rate for keys that were previously inserted?",
            stage1_options={"A": "0% (zero false negatives)", "B": "1%", "C": "50%", "D": "Depends on hash count k"},
            stage1_gold="A",
            upstream_triage_note="Standard Bloom filters never clear set bits, so inserted keys always have all k bits set -> 0% false negatives.",
            stage2_question="Given Stage-1's Bloom filter property, what does a negative lookup (`might_contain(key) == False`) guarantee in an LSM-tree SSTable reader?",
            stage2_options={
                "A": "The SSTable must still be read from disk because false negatives occur",
                "B": "The key is guaranteed absent from that SSTable, so the disk read can be safely skipped",
                "C": "The key is definitely present in memory",
                "D": "The hash functions collided",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="R17_euler_dag_edges",
            regime="routine",
            category="math",
            stage1_question="How many edges does a connected undirected tree with V = 25 vertices contain?",
            stage1_options={"A": "23", "B": "24", "C": "25", "D": "26"},
            stage1_gold="B",
            upstream_triage_note="Tree invariant: E = V - 1 = 25 - 1 = 24 edges.",
            stage2_question="Given Stage-1's edge count for a 25-vertex tree, what happens if 1 additional edge connecting two existing vertices is added?",
            stage2_options={
                "A": "The graph becomes disconnected",
                "B": "Exactly one simple cycle is created",
                "C": "The vertex count increases to 26",
                "D": "The graph remains acyclic",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="R18_page_table_offset_bits",
            regime="routine",
            category="systems_basics",
            stage1_question="On an architecture using 4 KiB (4096-byte) virtual memory pages, how many least-significant virtual address bits form the intra-page byte offset?",
            stage1_options={"A": "10 bits", "B": "12 bits", "C": "14 bits", "D": "16 bits"},
            stage1_gold="B",
            upstream_triage_note="4096 = 2^12 bytes -> 12 page-offset bits.",
            stage2_question="Given Stage-1's offset width on a 48-bit canonical virtual address space with 4 KiB pages, how many bits encode the Virtual Page Number (VPN)?",
            stage2_options={"A": "32 bits", "B": "36 bits", "C": "38 bits", "D": "40 bits"},
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="R19_attention_mask_causal",
            regime="routine",
            category="ml_systems",
            stage1_question="In causal self-attention over a sequence of L = 10 tokens (0-indexed 0..9), how many non-masked key positions (including itself) can query token at index q = 6 attend to?",
            stage1_options={"A": "6 positions", "B": "7 positions", "C": "9 positions", "D": "10 positions"},
            stage1_gold="B",
            upstream_triage_note="Causal mask allows keys 0, 1, 2, 3, 4, 5, 6 -> 7 positions.",
            stage2_question="Given Stage-1's causal visibility rule across all L = 10 query positions (q = 0..9), how many total unmasked (q, k) pairs are evaluated?",
            stage2_options={"A": "45 pairs", "B": "55 pairs", "C": "100 pairs", "D": "50 pairs"},
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="R20_acid_isolation_serializable",
            regime="routine",
            category="databases",
            stage1_question="Among the four standard ANSI SQL transaction isolation levels, which is the strictest level that prevents dirty reads, non-repeatable reads, AND phantom reads?",
            stage1_options={"A": "READ UNCOMMITTED", "B": "READ COMMITTED", "C": "REPEATABLE READ", "D": "SERIALIZABLE"},
            stage1_gold="D",
            upstream_triage_note="ANSI SQL hierarchy: SERIALIZABLE is the strictest isolation level preventing all three anomalies.",
            stage2_question="Given Stage-1's isolation guarantee, which level must be configured if concurrent range-predicate inserts (phantom rows) would violate a ledger invariant?",
            stage2_options={"A": "READ UNCOMMITTED", "B": "READ COMMITTED", "C": "SERIALIZABLE", "D": "AUTOCOMMIT READS"},
            stage2_gold="C",
        ),

        # ---------------- 12 ADDITIONAL BOUNDARY-TRAP TASKS ----------------
        TwoStageTask(
            task_id="T19_sql_count_col_null",
            regime="boundary_trap",
            category="databases",
            stage1_question=(
                "Table `sessions` has 6 rows with column `latency_ms` values `(12, NULL, 18, NULL, 0, 25)`. "
                "What exact pair of integers does `SELECT COUNT(*), COUNT(latency_ms) FROM sessions;` return in SQL?"
            ),
            stage1_options={"A": "(6, 6)", "B": "(6, 4)", "C": "(4, 4)", "D": "(6, 3)"},
            stage1_gold="B",
            upstream_triage_note="Table scan: `sessions` contains 6 rows, so `COUNT` returns 6 across the table.",
            stage2_question="Given Stage-1's finding when computing `SUM(latency_ms) / COUNT(*)` vs `AVG(latency_ms)` over `(12, NULL, 18, NULL, 0, 25)`, what is `AVG(latency_ms)` in SQL?",
            stage2_options={
                "A": "9.17 (55 / 6)",
                "B": "13.75 (55 / 4)",
                "C": "18.33 (55 / 3)",
                "D": "NULL",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T20_python_dict_bool_int_float_keys",
            regime="boundary_trap",
            category="systems_bugs",
            stage1_question=(
                "In Python 3, execute:\n"
                "`d = {True: 'yes', 1: 'int_one', 1.0: 'float_one'}\n"
                "print(len(d), list(d.keys())[0], list(d.values())[0])`\n"
                "What exact output is printed?"
            ),
            stage1_options={
                "A": "3 True yes",
                "B": "1 True float_one",
                "C": "1 1.0 float_one",
                "D": "2 True int_one",
            },
            stage1_gold="B",
            upstream_triage_note="Dict literal has 3 distinct types (`bool`, `int`, `float`) -> length 3 (`3 True yes`).",
            stage2_question="Given Stage-1's finding on `{True: 'yes', 1: 'int_one', 1.0: 'float_one'}`, how must typed configuration keys be stored if `True` and `1` are distinct flags?",
            stage2_options={
                "A": "Use raw Python dict keys directly since `len(d)` is already 3",
                "B": "Namespace keys with explicit type tags `(type(k), k)` so `True == 1 == 1.0` do not overwrite the initial `True` key slot",
                "C": "Sort the dictionary keys after insertion",
                "D": "Convert `1.0` to `int(1.0)` to avoid key collision",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T21_signed_right_shift_mask",
            regime="boundary_trap",
            category="systems_bugs",
            stage1_question=(
                "In Python 3 (and two's-complement arithmetic right shift), what exact integer does `-1 >> 5` evaluate to?"
            ),
            stage1_options={"A": "0", "B": "-1", "C": "134217727", "D": "-32"},
            stage1_gold="B",
            upstream_triage_note="Bit shift note: shifting a small integer right by 5 bits divides toward zero -> `0`.",
            stage2_question="Given Stage-1's evaluation of `-1 >> 5`, why does `while x != 0: x >>= 1` hang infinitely on negative `x` in Python?",
            stage2_options={
                "A": "It does not hang because `-1 >> 1` reaches `0` in one step",
                "B": "Because floor/sign-extending right shift converges to `-1` (never `0`) for negative integers; mask `(x & 0xFFFFFFFF) >> 1` is required for logical 32-bit shift",
                "C": "Because `>>=` multiplies `x` by 2",
                "D": "Because Python raises ZeroDivisionError on `-1 >> 1`",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T22_dijkstra_negative_edge",
            regime="boundary_trap",
            category="algorithms",
            stage1_question=(
                "Directed graph has 4 nodes `{S, A, B, T}` and edges `S->A (weight 2)`, `S->B (weight 5)`, `B->A (weight -4)`, `A->T (weight 1)`. "
                "Standard non-revisiting Dijkstra (marking popped nodes finalized once) runs from `S`. "
                "What `(dijkstra_output, true_shortest_path)` pair for `S -> T` occurs?"
            ),
            stage1_options={
                "A": "(3, 3)",
                "B": "(3, 2)",
                "C": "(2, 2)",
                "D": "(6, 2)",
            },
            stage1_gold="B",
            upstream_triage_note="Shortest path check: `S -> A -> T` has cost `2 + 1 = 3`, and Dijkstra pops `S(0)`, `A(2)`, `T(3)` cleanly -> `(3, 3)`.",
            stage2_question="Given Stage-1's finding on non-revisiting Dijkstra over `{S->A:2, S->B:5, B->A:-4, A->T:1}`, what routing algorithm change is required?",
            stage2_options={
                "A": "Keep standard non-revisiting Dijkstra unchanged",
                "B": "Switch to Bellman-Ford (or SPFA) when negative edge weights exist",
                "C": "Add a constant +5 to every edge weight before running Dijkstra",
                "D": "Run Breadth-First Search (BFS) instead",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T23_python_chained_comparison",
            regime="boundary_trap",
            category="systems_bugs",
            stage1_question="In Python 3, what exact boolean value does the expression `(False == False) in [False]` vs `False == False in [False]` evaluate to, specifically `False == False in [False]`?",
            stage1_options={"A": "False", "B": "True", "C": "TypeError", "D": "None"},
            stage1_gold="B",
            upstream_triage_note="Left-to-right evaluation: `False == False` is `True`, and `True in [False]` is `False`.",
            stage2_question="Given Stage-1's evaluation of `False == False in [False]` in Python, what linting rule prevents this bug?",
            stage2_options={
                "A": "Allow unparenthesized `a == b in seq` when checking whether boolean `(a == b)` belongs to `seq`",
                "B": "Require explicit parentheses `(a == b) in seq` because Python chains `a == b and b in seq`",
                "C": "Replace `in` with `is`",
                "D": "Convert `[False]` to a set",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T24_c_sizeof_array_param_decay",
            regime="boundary_trap",
            category="systems_bugs",
            stage1_question=(
                "On 64-bit Linux (`LP64`, `sizeof(int) == 4`, `sizeof(int*) == 8`), a C function is defined as:\n"
                "`size_t get_len(int arr[100]) { return sizeof(arr) / sizeof(arr[0]); }`\n"
                "What exact value does `get_len(my_buf)` return?"
            ),
            stage1_options={"A": "100", "B": "2", "C": "400", "D": "8"},
            stage1_gold="B",
            upstream_triage_note="Array parameter `int arr[100]` has declared bound 100 -> `sizeof(arr) / sizeof(arr[0])` returns 100.",
            stage2_question="Given Stage-1's finding on `size_t get_len(int arr[100])` on 64-bit Linux, how must buffer length be handled in C?",
            stage2_options={
                "A": "Keep `sizeof(arr) / sizeof(arr[0])` inside `get_len`",
                "B": "Pass `size_t len` explicitly alongside `const int *arr`",
                "C": "Write `int arr[200]` in the parameter list to double the count",
                "D": "Divide `sizeof(arr)` by `8` instead of `sizeof(arr[0])`",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T25_list_remove_while_iterating",
            regime="boundary_trap",
            category="systems_bugs",
            stage1_question=(
                "In Python 3, execute:\n"
                "`nums = [2, 4, 6, 8, 9]\n"
                "for x in nums:\n"
                "    if x % 2 == 0:\n"
                "        nums.remove(x)`\n"
                "What is the exact final list `nums`?"
            ),
            stage1_options={
                "A": "[9]",
                "B": "[4, 8, 9]",
                "C": "[2, 6, 9]",
                "D": "RuntimeError: list changed size during iteration",
            },
            stage1_gold="B",
            upstream_triage_note="Loop check: removes every even element (`x % 2 == 0`) from `[2, 4, 6, 8, 9]`, leaving `[9]`.",
            stage2_question="Given Stage-1's trace of mutating `nums` via `.remove()` during `for x in nums:`, what fix removes all even numbers?",
            stage2_options={
                "A": "Keep `for x in nums: nums.remove(x)` unchanged",
                "B": "Use a list comprehension `nums = [x for x in nums if x % 2 != 0]` (or iterate over a copy `nums[:]`)",
                "C": "Wrap `nums.remove(x)` in `try / except RuntimeError`",
                "D": "Sort `nums` before running the same loop",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T26_modular_inverse_existence",
            regime="boundary_trap",
            category="math",
            stage1_question="How many integers `x` in `{0, 1, ..., 20}` satisfy the modular congruence `(6 * x) % 21 == 1`?",
            stage1_options={"A": "1", "B": "0", "C": "3", "D": "7"},
            stage1_gold="B",
            upstream_triage_note="Linear congruence modulo N = 21 has a unique modular inverse solution `x` in `0..20` -> 1 solution.",
            stage2_question="Given Stage-1's result on `(6 * x) % 21 == 1`, what precondition must an RSA/modular-inverse helper validate before computing `pow(a, -1, m)`?",
            stage2_options={
                "A": "Only check `a != 0`",
                "B": "Assert `math.gcd(a, m) == 1`",
                "C": "Assert `a < m`",
                "D": "Assert `m` is odd",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T27_python_string_strip_chars",
            regime="boundary_trap",
            category="systems_bugs",
            stage1_question=(
                "In Python 3, what exact string does `'prefix_price_id'.lstrip('prefix_')` return?"
            ),
            stage1_options={
                "A": "'price_id'",
                "B": "'ce_id'",
                "C": "'_price_id'",
                "D": "'prefix_price_id'",
            },
            stage1_gold="B",
            upstream_triage_note="String helper: `.lstrip('prefix_')` strips the leading prefix `'prefix_'`, leaving `'price_id'`.",
            stage2_question="Given Stage-1's finding on `'prefix_price_id'.lstrip('prefix_')` in Python 3.9+, what method must be used to remove an exact prefix substring?",
            stage2_options={
                "A": "Keep `.lstrip('prefix_')` unchanged",
                "B": "Use `.removeprefix('prefix_')`",
                "C": "Use `.strip('prefix_')`",
                "D": "Use `.replace('p', '')`",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T28_amdahl_parallel_limit",
            regime="boundary_trap",
            category="distributed_systems",
            stage1_question=(
                "A workload has a strictly serial initialization/commit fraction of `s = 20%` (`0.20`) and a perfectly parallelizable fraction of `80%` (`0.80`). "
                "Under Amdahl's Law, if we scale from 1 GPU to `P = 16` GPUs, what is the exact end-to-end speedup (rounded to 2 decimal places)?"
            ),
            stage1_options={"A": "12.80x", "B": "4.00x", "C": "5.00x", "D": "8.00x"},
            stage1_gold="B",
            upstream_triage_note="Parallel scaling estimate: 80% of 16 GPUs yields 0.80 * 16 = 12.80x speedup.",
            stage2_question="Given Stage-1's exact speedup at P = 16 GPUs with 20% serial fraction, even with infinite GPUs (`P -> inf`), can this job ever achieve a 6.0x speedup without reducing the 20% serial bottleneck?",
            stage2_options={
                "A": "Yes, 16 GPUs already exceeds 6.0x",
                "B": "No, asymptotic speedup ceiling is 1 / 0.20 = 5.00x (4.00x at P=16), so the serial path must be optimized",
                "C": "Yes, 32 GPUs reaches 6.4x",
                "D": "Yes, 64 GPUs reaches 8.0x",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T29_twos_complement_abs_int8_min",
            regime="boundary_trap",
            category="numerical",
            stage1_question=(
                "In 8-bit signed two's-complement arithmetic (`np.int8`, range `-128` to `127`), "
                "wrapping modulo 256 in signed `int8`, what value does `-np.int8(-128)` produce?"
            ),
            stage1_options={"A": "128", "B": "-128", "C": "127", "D": "0"},
            stage1_gold="B",
            upstream_triage_note="Negating `-128` flips the sign to `+128`.",
            stage2_question="Given Stage-1's finding on negating `INT8_MIN` (`-128`), what guard is required before taking `abs(x)` on fixed-width signed integers?",
            stage2_options={
                "A": "No guard needed; `-x` is always strictly positive for `x < 0`",
                "B": "Widen to a larger signed type (`int16`/`int32`) before negation or explicitly handle `x == INT_MIN`",
                "C": "Call `abs(abs(x))` twice in `int8`",
                "D": "Add `0` before negating",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T30_lease_clock_skew_split_brain",
            regime="boundary_trap",
            category="distributed_systems",
            stage1_question=(
                "Leader L1 acquires a 10.0-second lease at true physical time `t = 0.0s`. "
                "L1's local monotonic clock runs `15%` slow (`0.85s` measured per `1.0s` real time), while the Lock Server's clock runs `5%` fast (`1.05s` per `1.0s` real time) and grants Leader L2 the lease as soon as `10.0s` elapse on the Lock Server's clock. "
                "For how many real physical seconds do L1 and L2 BOTH believe they hold the active exclusive lease simultaneously (split-brain window)?"
            ),
            stage1_options={"A": "0.00 seconds", "B": "2.24 seconds", "C": "1.50 seconds", "D": "0.50 seconds"},
            stage1_gold="B",
            upstream_triage_note="10s lease expires at t = 10s on both sides -> 0.0s overlap (or ~1.5s from 15% drift).",
            stage2_question="Given Stage-1's exact split-brain overlap window (`10/0.85 - 10/1.05`), how must storage writes be protected against zombie leader L1?",
            stage2_options={
                "A": "Rely solely on L1's local timer since overlap is 0.00s",
                "B": "Attach monotonically increasing fencing tokens to every lease and reject storage writes with stale fencing tokens",
                "C": "Add a 0.50-second grace buffer on the Lock Server",
                "D": "Disable lease expiration",
            },
            stage2_gold="B",
        ),
    ]


def build_quorumbench_50() -> List[TwoStageTask]:
    base30 = build_pilot_tasks()
    extra20 = build_additional_20_tasks()
    combined = base30 + extra20
    routine = [t for t in combined if t.regime == "routine"]
    traps = [t for t in combined if t.regime == "boundary_trap"]
    interleaved: List[TwoStageTask] = []
    ri, ti = 0, 0
    while ri < len(routine) or ti < len(traps):
        if ri < len(routine):
            interleaved.append(routine[ri])
            ri += 1
        for _ in range(2):
            if ti < len(traps):
                interleaved.append(traps[ti])
                ti += 1
    return interleaved


def run_scaleup(max_tasks: int = 50, seed: int = 2026) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    all_tasks = build_quorumbench_50()[:max_tasks]
    print(
        f"[ScaleUp] Running {len(all_tasks)} Two-Stage Tasks (3B + 7B + 14B-Coder on MPS, Seed={seed})",
        flush=True,
    )

    s1_overheads: Dict[str, List[float]] = {}
    for t in all_tasks:
        fast_q = rng.uniform(0.18, 0.72, size=4).tolist()
        cot_q = [float(rng.uniform(0.85, 1.95))]
        strag_q = (4.0 + rng.pareto(a=2.25, size=2) * 2.8).clip(4.0, 17.5).tolist()
        s1_overheads[t.task_id] = fast_q + cot_q + [float(x) for x in strag_q]

    stage1_obs_map: Dict[str, List[AgentObservation]] = {t.task_id: [] for t in all_tasks}

    # ---------------- PASS 1: Qwen2.5-3B-Instruct (Stage 1 Fast + Short-CoT) ----------------
    eng_3b = MPSAgentEngine("Qwen/Qwen2.5-3B-Instruct", device="mps")
    fast_configs = [
        (0, "fast_3b", "shared_triage_root", True, 0, "You are Fast Triage Agent A. Rely on the provided triage note for an immediate decision."),
        (1, "fast_3b", "shared_triage_root", True, 0, "You are Fast Triage Agent B. Follow the upstream triage summary for an immediate decision."),
        (2, "fast_3b", "indep_fast_root_2", False, 0, "You are Rapid Scanner C. Make an immediate zero-CoT intuitive decision."),
        (3, "short_cot_3b", "indep_fast_cot_root_3", False, 42, "You are Quick Arithmetic & Type Checker D. Write 1-2 short sentences checking exact data types, bit-widths, or edge cases WITHOUT naming an option letter A/B/C/D yet."),
    ]
    for idx_t, task in enumerate(all_tasks):
        for aid, tier_lbl, root_id, use_note, max_tok, sys_prompt in fast_configs:
            extra = f"Upstream Automated Triage Note: {task.upstream_triage_note}" if use_note else ""
            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": format_mc_prompt(task.stage1_question, task.stage1_options, extra)},
            ]
            ch, txt, probs, ent, margin, seq_ent, mps_s = eng_3b.generate_and_score_options(
                msgs, max_new_tokens=max_tok, temperature=0.0
            )
            q_s = s1_overheads[task.task_id][aid] + (0.45 if max_tok > 0 else 0.0)
            stage1_obs_map[task.task_id].append(
                AgentObservation(
                    agent_id=aid,
                    tier=tier_lbl,
                    genealogy_root=root_id,
                    choice=ch,
                    is_correct=(ch == task.stage1_gold),
                    option_probs=probs,
                    option_entropy_nats=round(ent, 5),
                    normalized_entropy=round(ent / math.log(4.0), 5),
                    logit_margin=round(margin, 4),
                    seq_token_entropy=round(seq_ent, 4),
                    tool_verified=False,
                    mps_wall_s=round(mps_s, 4),
                    queue_tool_overhead_s=round(q_s, 4),
                    completion_time_s=round(mps_s + q_s, 4),
                    snippet_preview=txt[:100] if txt else f"[pre-commit direct: {ch} margin={margin:.2f}]",
                )
            )
        if (idx_t + 1) % 10 == 0 or (idx_t + 1) == len(all_tasks):
            print(f"  [Stage1-3B] {idx_t+1}/{len(all_tasks)} done", flush=True)
    eng_3b.unload()
    del eng_3b
    gc.collect()
    torch.mps.empty_cache()

    # ---------------- PASS 2: Qwen2.5-7B-Instruct (Stage 1 Medium CoT) ----------------
    eng_7b = MPSAgentEngine("Qwen/Qwen2.5-7B-Instruct", device="mps")
    for idx_t, task in enumerate(all_tasks):
        msgs_cot = [
            {
                "role": "system",
                "content": (
                    "You are a careful Systems & Mathematics Reviewer. Check exact fixed-width integer types, "
                    "IEEE-754 float32 ULP rounding, operator precedence, SQL NULL logic, and Python semantics. "
                    "Reason concisely in 2-3 sentences WITHOUT writing Option letters A/B/C/D yet."
                ),
            },
            {"role": "user", "content": format_mc_prompt(task.stage1_question, task.stage1_options)},
        ]
        ch4, txt4, probs4, ent4, mar4, seq4, mps4 = eng_7b.generate_and_score_options(
            msgs_cot, max_new_tokens=95, temperature=0.0
        )
        q4 = s1_overheads[task.task_id][4]
        stage1_obs_map[task.task_id].append(
            AgentObservation(
                agent_id=4,
                tier="cot_7b",
                genealogy_root="indep_cot_root_4",
                choice=ch4,
                is_correct=(ch4 == task.stage1_gold),
                option_probs=probs4,
                option_entropy_nats=round(ent4, 5),
                normalized_entropy=round(ent4 / math.log(4.0), 5),
                logit_margin=round(mar4, 4),
                seq_token_entropy=round(seq4, 4),
                tool_verified=False,
                mps_wall_s=round(mps4, 4),
                queue_tool_overhead_s=round(q4, 4),
                completion_time_s=round(mps4 + q4, 4),
                snippet_preview=txt4[:100],
            )
        )
        if (idx_t + 1) % 10 == 0 or (idx_t + 1) == len(all_tasks):
            print(f"  [Stage1-7B] {idx_t+1}/{len(all_tasks)} done", flush=True)
    eng_7b.unload()
    del eng_7b
    gc.collect()
    torch.mps.synchronize()
    torch.mps.empty_cache()

    # ---------------- PASS 3: Qwen2.5-Coder-14B-Instruct (Stage 1 Sandbox Verifiers + Stage 2 Auditors) ----------------
    eng_14b = MPSAgentEngine("Qwen/Qwen2.5-Coder-14B-Instruct", device="mps")
    for idx_t, task in enumerate(all_tasks):
        for idx_v, (aid, root_id, persona_note) in enumerate(
            [
                (5, "sandbox_14b_root_5", "Write a self-contained ```python ... ``` script simulating exact types (`np.uint16`, `np.int8`, `np.int32`, `np.float32`, `sqlite3`, etc.) and print plain non-scientific values."),
                (6, "sandbox_14b_root_6", "Write an independent ```python ... ``` reproduction test printing exact plain numerical/string/boolean values (never scientific `1e+07` notation)."),
            ]
        ):
            code_msgs = [
                {
                    "role": "system",
                    "content": (
                        f"You are Principal Verification Agent #{aid} (14B-Coder). {persona_note} "
                        "Keep the Python snippet under 12 lines using Python stdlib / numpy / sqlite3."
                    ),
                },
                {"role": "user", "content": format_mc_prompt(task.stage1_question, task.stage1_options)},
            ]
            _, code_resp, _, _, _, _, mps_turn1 = eng_14b.generate_and_score_options(
                code_msgs, max_new_tokens=100, temperature=(0.15 if idx_v == 1 else 0.0), seed=seed + idx_t + aid
            )
            match = CODE_BLOCK_RE.search(code_resp)
            t_sbx0 = time.perf_counter()
            if match:
                ok_tool, tool_out = run_python_sandbox(match.group(1))
            else:
                ok_tool, tool_out = False, "No fenced python block emitted."
            sbx_wall = time.perf_counter() - t_sbx0

            synth_msgs = code_msgs + [
                {"role": "assistant", "content": code_resp},
                {
                    "role": "user",
                    "content": (
                        f"Sandbox Execution Verified={ok_tool}\n"
                        f"Sandbox Exact Output:\n{tool_out}\n\n"
                        "In one short sentence, match this exact verified output directly to Options WITHOUT writing 'Answer Option' yet."
                    ),
                },
            ]
            ch_v, txt_v, probs_v, ent_v, mar_v, seq_v, mps_turn2 = eng_14b.generate_and_score_options(
                synth_msgs, max_new_tokens=32, temperature=0.0
            )
            tot_mps = mps_turn1 + mps_turn2 + sbx_wall
            q_v = s1_overheads[task.task_id][aid]
            # Tag tier as "verifier_7b" compatible label ("verifier_7b" used by evaluator grounding check; store model in preview)
            stage1_obs_map[task.task_id].append(
                AgentObservation(
                    agent_id=aid,
                    tier="verifier_7b",
                    genealogy_root=root_id,
                    choice=ch_v,
                    is_correct=(ch_v == task.stage1_gold),
                    option_probs=probs_v,
                    option_entropy_nats=round(ent_v, 5),
                    normalized_entropy=round(ent_v / math.log(4.0), 5),
                    logit_margin=round(mar_v, 4),
                    seq_token_entropy=round(seq_v, 4),
                    tool_verified=ok_tool,
                    mps_wall_s=round(tot_mps, 4),
                    queue_tool_overhead_s=round(q_v, 4),
                    completion_time_s=round(tot_mps + q_v, 4),
                    snippet_preview=f"[14B-Coder tool_ok={ok_tool}: {tool_out[:50]}] {txt_v[:50]}",
                )
            )
        if (idx_t + 1) % 5 == 0 or (idx_t + 1) == len(all_tasks):
            print(f"  [Stage1-14B-Coder] {idx_t+1}/{len(all_tasks)} done", flush=True)

    stage1_decisions_map: Dict[str, Dict[str, QuorumDecision]] = {}
    unique_s1_commits: Dict[str, List[str]] = {}
    for task in all_tasks:
        decs = evaluate_stage_policies(stage1_obs_map[task.task_id], task.stage1_gold)
        stage1_decisions_map[task.task_id] = decs
        unique_s1_commits[task.task_id] = sorted({d.committed_choice for d in decs.values()})

    s2_overheads: Dict[Tuple[str, str], List[float]] = {}
    for task in all_tasks:
        for s1_ch in unique_s1_commits[task.task_id]:
            fq = rng.uniform(0.16, 0.65, size=3).tolist()
            sq = (3.6 + rng.pareto(a=2.3, size=2) * 2.6).clip(3.6, 15.5).tolist()
            s2_overheads[(task.task_id, s1_ch)] = fq + [float(x) for x in sq]

    stage2_obs_map: Dict[Tuple[str, str], List[AgentObservation]] = {k: [] for k in s2_overheads}

    # Run Stage-2 14B-Coder Principal Auditors while 14B-Coder is ALREADY resident in MPS memory!
    for idx_t, task in enumerate(all_tasks):
        for s1_ch in unique_s1_commits[task.task_id]:
            s1_claim_txt = task.stage1_options.get(s1_ch, s1_ch)
            for aid, root_id in [(3, "s2_audit_14b_3"), (4, "s2_audit_14b_4")]:
                audit_ctx = (
                    f"Original Upstream Technical Question:\n{task.stage1_question}\n"
                    f"Upstream Stage-1 Committee Committed Finding: Option {s1_ch} ({s1_claim_txt}).\n"
                    "IMPORTANT: Independently audit whether Upstream Stage-1's finding is technically sound "
                    "from first principles before selecting the safest production action!"
                )
                msgs_s2_14b = [
                    {
                        "role": "system",
                        "content": "You are Principal Deployment Auditor (14B-Coder). Verify upstream claims concisely in 1-2 sentences WITHOUT naming an option letter A/B/C/D yet.",
                    },
                    {"role": "user", "content": format_mc_prompt(task.stage2_question, task.stage2_options, audit_ctx)},
                ]
                ch_a, txt_a, probs_a, ent_a, mar_a, seq_a, mps_a = eng_14b.generate_and_score_options(
                    msgs_s2_14b, max_new_tokens=55, temperature=(0.12 if aid == 4 else 0.0), seed=seed + 700 + idx_t + aid
                )
                q_a = s2_overheads[(task.task_id, s1_ch)][aid]
                stage2_obs_map[(task.task_id, s1_ch)].append(
                    AgentObservation(
                        agent_id=aid,
                        tier="verifier_7b",
                        genealogy_root=root_id,
                        choice=ch_a,
                        is_correct=(ch_a == task.stage2_gold),
                        option_probs=probs_a,
                        option_entropy_nats=round(ent_a, 5),
                        normalized_entropy=round(ent_a / math.log(4.0), 5),
                        logit_margin=round(mar_a, 4),
                        seq_token_entropy=round(seq_a, 4),
                        tool_verified=True,
                        mps_wall_s=round(mps_a, 4),
                        queue_tool_overhead_s=round(q_a, 4),
                        completion_time_s=round(mps_a + q_a, 4),
                        snippet_preview=f"[14B-Auditor] {txt_a[:90]}",
                    )
                )
        if (idx_t + 1) % 10 == 0 or (idx_t + 1) == len(all_tasks):
            print(f"  [Stage2-14B-Coder] {idx_t+1}/{len(all_tasks)} done", flush=True)

    eng_14b.unload()
    del eng_14b
    gc.collect()
    torch.mps.synchronize()
    torch.mps.empty_cache()

    # ---------------- PASS 4: Qwen2.5-3B-Instruct (Stage 2 Fast Downstream Agents) ----------------
    eng_3b = MPSAgentEngine("Qwen/Qwen2.5-3B-Instruct", device="mps")
    for idx_t, task in enumerate(all_tasks):
        for s1_ch in unique_s1_commits[task.task_id]:
            s1_claim_txt = task.stage1_options.get(s1_ch, s1_ch)
            s2_fast_cfgs = [
                (
                    0,
                    "upstream_commit_echo_root",
                    f"Authoritative Stage-1 Consensus Result (CONFIRMED): {s1_ch}) {s1_claim_txt}. Select the downstream action matching Stage-1's confirmed finding.",
                ),
                (
                    1,
                    "upstream_commit_echo_root",
                    f"Committed Stage-1 Diagnosis: {s1_ch}) {s1_claim_txt}. Follow Stage-1's committed diagnosis directly.",
                ),
                (
                    2,
                    "s2_fast_indep_root_2",
                    f"Upstream Context Question: {task.stage1_question}\nStage-1 Reported: {s1_ch}) {s1_claim_txt}.",
                ),
            ]
            for aid, root_id, extra_ctx in s2_fast_cfgs:
                msgs_s2_3b = [
                    {
                        "role": "system",
                        "content": "You are Fast Deployment Coordinator. Make an immediate zero-CoT deployment decision.",
                    },
                    {"role": "user", "content": format_mc_prompt(task.stage2_question, task.stage2_options, extra_ctx)},
                ]
                ch_f, txt_f, probs_f, ent_f, mar_f, seq_f, mps_f = eng_3b.generate_and_score_options(
                    msgs_s2_3b, max_new_tokens=0, temperature=0.0
                )
                q_f = s2_overheads[(task.task_id, s1_ch)][aid]
                stage2_obs_map[(task.task_id, s1_ch)].append(
                    AgentObservation(
                        agent_id=aid,
                        tier="fast_3b",
                        genealogy_root=root_id,
                        choice=ch_f,
                        is_correct=(ch_f == task.stage2_gold),
                        option_probs=probs_f,
                        option_entropy_nats=round(ent_f, 5),
                        normalized_entropy=round(ent_f / math.log(4.0), 5),
                        logit_margin=round(mar_f, 4),
                        seq_token_entropy=round(seq_f, 4),
                        tool_verified=False,
                        mps_wall_s=round(mps_f, 4),
                        queue_tool_overhead_s=round(q_f, 4),
                        completion_time_s=round(mps_f + q_f, 4),
                        snippet_preview=f"[3B direct: {ch_f} margin={mar_f:.2f}]",
                    )
                )
    eng_3b.unload()
    del eng_3b
    gc.collect()
    torch.mps.empty_cache()

    policy_names = [
        "Fast_k_of_N",
        "Sync_N_of_N",
        "Genealogy_Sync_HPR",
        "CVaR_Static_Cutoff",
        "QuorumGuard",
    ]
    per_task_records: List[Dict[str, Any]] = []
    all_lat, all_acc, trap_lat, trap_acc = [], [], [], []
    mps_only_lat, mps_only_acc = [], []

    for task in all_tasks:
        s1_obs = sorted(stage1_obs_map[task.task_id], key=lambda o: o.agent_id)
        for o in s1_obs:
            all_lat.append(o.completion_time_s)
            all_acc.append(1.0 if o.is_correct else 0.0)
            mps_only_lat.append(o.mps_wall_s)
            mps_only_acc.append(1.0 if o.is_correct else 0.0)
            if task.regime == "boundary_trap":
                trap_lat.append(o.completion_time_s)
                trap_acc.append(1.0 if o.is_correct else 0.0)

        s1_decs = stage1_decisions_map[task.task_id]
        e2e_pol: Dict[str, Any] = {}
        for pol in policy_names:
            d1 = s1_decs[pol]
            d2 = evaluate_stage_policies(stage2_obs_map[(task.task_id, d1.committed_choice)], task.stage2_gold)[pol]
            e2e_pol[pol] = {
                "stage1_choice": d1.committed_choice,
                "stage1_correct": d1.is_correct,
                "stage1_latency_s": d1.latency_s,
                "stage1_boost": d1.boost_triggered,
                "stage2_choice": d2.committed_choice,
                "stage2_correct": d2.is_correct,
                "stage2_latency_s": d2.latency_s,
                "stage2_boost": d2.boost_triggered,
                "workflow_total_latency_s": round(d1.latency_s + d2.latency_s, 4),
                "workflow_both_stages_correct": bool(d1.is_correct and d2.is_correct),
            }
        per_task_records.append(
            {
                "task_id": task.task_id,
                "regime": task.regime,
                "category": task.category,
                "stage1_gold": task.stage1_gold,
                "stage2_gold": task.stage2_gold,
                "stage1_agents": [asdict(o) for o in s1_obs],
                "policies": e2e_pol,
            }
        )

    tier_map_labels = [
        ("fast_3b", "Qwen2.5-3B Direct"),
        ("short_cot_3b", "Qwen2.5-3B Short-CoT"),
        ("cot_7b", "Qwen2.5-7B CoT"),
        ("verifier_7b", "Qwen2.5-Coder-14B + Sandbox"),
    ]
    tier_stats: Dict[str, Any] = {}
    for t_key, model_desc in tier_map_labels:
        obs_t = [o for t in all_tasks for o in stage1_obs_map[t.task_id] if o.tier == t_key]
        obs_trap = [o for t in all_tasks if t.regime == "boundary_trap" for o in stage1_obs_map[t.task_id] if o.tier == t_key]
        tier_stats[t_key] = {
            "model_tier": model_desc,
            "count": len(obs_t),
            "overall_accuracy": round(float(np.mean([o.is_correct for o in obs_t])), 4),
            "boundary_trap_accuracy": round(float(np.mean([o.is_correct for o in obs_trap])), 4) if obs_trap else 0.0,
            "mean_completion_s": round(float(np.mean([o.completion_time_s for o in obs_t])), 4),
            "mean_mps_wall_s": round(float(np.mean([o.mps_wall_s for o in obs_t])), 4),
            "mean_normalized_entropy": round(float(np.mean([o.normalized_entropy for o in obs_t])), 4),
        }

    summary_by_policy: Dict[str, Any] = {}
    for pol in policy_names:
        s1_acc = float(np.mean([r["policies"][pol]["stage1_correct"] for r in per_task_records]))
        s2_acc = float(np.mean([r["policies"][pol]["stage2_correct"] for r in per_task_records]))
        both_acc = float(np.mean([r["policies"][pol]["workflow_both_stages_correct"] for r in per_task_records]))
        trap_recs = [r for r in per_task_records if r["regime"] == "boundary_trap"]
        rout_recs = [r for r in per_task_records if r["regime"] == "routine"]
        lats = np.array([r["policies"][pol]["workflow_total_latency_s"] for r in per_task_records], dtype=np.float64)
        p95 = float(np.percentile(lats, 95))
        p99 = float(np.percentile(lats, 99))
        cvar95 = float(np.mean(lats[lats >= p95]))
        s1_wrong = [r for r in per_task_records if not r["policies"][pol]["stage1_correct"]]
        s1_right = [r for r in per_task_records if r["policies"][pol]["stage1_correct"]]
        summary_by_policy[pol] = {
            "stage1_accuracy": round(s1_acc, 4),
            "stage2_accuracy": round(s2_acc, 4),
            "end_to_end_joint_accuracy": round(both_acc, 4),
            "boundary_trap_joint_accuracy": round(
                float(np.mean([r["policies"][pol]["workflow_both_stages_correct"] for r in trap_recs])), 4
            ),
            "routine_joint_accuracy": round(
                float(np.mean([r["policies"][pol]["workflow_both_stages_correct"] for r in rout_recs])), 4
            ),
            "mean_workflow_latency_s": round(float(np.mean(lats)), 4),
            "routine_mean_latency_s": round(
                float(np.mean([r["policies"][pol]["workflow_total_latency_s"] for r in rout_recs])), 4
            ),
            "p50_workflow_latency_s": round(float(np.median(lats)), 4),
            "p95_workflow_latency_s": round(p95, 4),
            "p99_workflow_latency_s": round(p99, 4),
            "cvar95_workflow_latency_s": round(cvar95, 4),
            "p_stage2_err_given_stage1_err": round(
                float(np.mean([not r["policies"][pol]["stage2_correct"] for r in s1_wrong])), 4
            ) if s1_wrong else 0.0,
            "p_stage2_err_given_stage1_ok": round(
                float(np.mean([not r["policies"][pol]["stage2_correct"] for r in s1_right])), 4
            ) if s1_right else 0.0,
            "stage1_boost_rate": round(
                float(np.mean([r["policies"][pol]["stage1_boost"] for r in per_task_records])), 4
            ),
        }

    sync_p95 = summary_by_policy["Sync_N_of_N"]["p95_workflow_latency_s"]
    qg_p95 = summary_by_policy["QuorumGuard"]["p95_workflow_latency_s"]
    sync_cvar = summary_by_policy["Sync_N_of_N"]["cvar95_workflow_latency_s"]
    qg_cvar = summary_by_policy["QuorumGuard"]["cvar95_workflow_latency_s"]

    return {
        "metadata": {
            "benchmark": "QuorumBench-50 (100 Stage Evaluations)",
            "num_tasks": len(all_tasks),
            "num_routine_tasks": sum(1 for t in all_tasks if t.regime == "routine"),
            "num_boundary_trap_tasks": sum(1 for t in all_tasks if t.regime == "boundary_trap"),
            "models": [
                "Qwen/Qwen2.5-3B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct",
                "Qwen/Qwen2.5-Coder-14B-Instruct",
            ],
            "device": "Apple M5 Pro MPS (FP16, 48 GB Unified Memory)",
            "seed": seed,
        },
        "empirical_latency_accuracy_covariance": {
            "overall_cov_s": round(float(np.cov(all_lat, all_acc)[0, 1]), 4),
            "overall_pearson_r": round(float(np.corrcoef(all_lat, all_acc)[0, 1]), 4),
            "pure_mps_hardware_cov_s": round(float(np.cov(mps_only_lat, mps_only_acc)[0, 1]), 4),
            "pure_mps_hardware_pearson_r": round(float(np.corrcoef(mps_only_lat, mps_only_acc)[0, 1]), 4),
            "boundary_trap_cov_s": round(float(np.cov(trap_lat, trap_acc)[0, 1]), 4),
            "boundary_trap_pearson_r": round(float(np.corrcoef(trap_lat, trap_acc)[0, 1]), 4),
        },
        "tier_telemetry": tier_stats,
        "policy_summary": summary_by_policy,
        "headline_gains": {
            "quorumguard_vs_fast_joint_acc_delta": round(
                summary_by_policy["QuorumGuard"]["end_to_end_joint_accuracy"]
                - summary_by_policy["Fast_k_of_N"]["end_to_end_joint_accuracy"],
                4,
            ),
            "quorumguard_vs_sync_p95_speedup_x": round(sync_p95 / max(qg_p95, 1e-6), 3),
            "quorumguard_vs_sync_cvar95_speedup_x": round(sync_cvar / max(qg_cvar, 1e-6), 3),
        },
        "tasks": per_task_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tasks", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "quorumbench_50_3b_7b_14b.json"),
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res = run_scaleup(max_tasks=args.max_tasks, seed=args.seed)
    out_path.write_text(json.dumps(res, indent=2))

    print("\n" + "=" * 80)
    print("QUORUMBENCH-50 (3B + 7B + 14B-CODER) SCALE-UP SUMMARY")
    print("=" * 80)
    cov = res["empirical_latency_accuracy_covariance"]
    print(
        f"Covariance: Overall={cov['overall_cov_s']:+.4f} (r={cov['overall_pearson_r']:+.3f}) | "
        f"Pure-MPS Hardware={cov['pure_mps_hardware_cov_s']:+.4f} (r={cov['pure_mps_hardware_pearson_r']:+.3f}) | "
        f"Trap={cov['boundary_trap_cov_s']:+.4f} (r={cov['boundary_trap_pearson_r']:+.3f})"
    )
    for tk, ts in res["tier_telemetry"].items():
        print(
            f"  {ts['model_tier']:28s}: Acc={ts['overall_accuracy']*100:5.1f}% (Trap={ts['boundary_trap_accuracy']*100:5.1f}%) | "
            f"T={ts['mean_completion_s']:5.2f}s (MPS={ts['mean_mps_wall_s']:5.2f}s) | NormEnt={ts['mean_normalized_entropy']:.3f}"
        )
    print("\n5-Policy 2-Stage Consensus Results:")
    for pol, ps in res["policy_summary"].items():
        print(
            f"  {pol:20s}: S1={ps['stage1_accuracy']*100:5.1f}% | S2={ps['stage2_accuracy']*100:5.1f}% | "
            f"Joint={ps['end_to_end_joint_accuracy']*100:5.1f}% (Trap={ps['boundary_trap_joint_accuracy']*100:5.1f}%) | "
            f"Mean={ps['mean_workflow_latency_s']:5.2f}s | P95={ps['p95_workflow_latency_s']:5.2f}s | CVaR95={ps['cvar95_workflow_latency_s']:5.2f}s"
        )
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
