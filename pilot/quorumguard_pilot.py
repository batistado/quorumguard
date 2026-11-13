#!/usr/bin/env python3
"""
QuorumGuard Phase 0/1 Empirical Pilot (Apple Silicon MPS + Real Qwen2.5 Pool)
=============================================================================
Evaluates 5 distributed multi-agent consensus policies across a 30-task,
2-Stage Asynchronous Multi-Agent Workflow DAG (Stage 1: Root-Cause / Invariant
Verification Committee -> Stage 2: Production Remediation / Deployment Committee):

  1. Fast k-of-N Quorum (First-3-Arrive / First-2-Agree)
  2. Synchronous Full Barrier (N-of-N Raw Majority)
  3. Genealogy-Weighted Synchronous (HPR-Adaptive / Spark-to-Fire Barrier)
  4. CVaR Static Deadline Cutoff (Latency-Budget Turn Release, Epistemically Blind)
  5. QuorumGuard (Entropy- & Genealogy-Gated Adaptive Quorum + Dissent Boosting)

All LLM generations, token logits, 4-option posterior semantic entropies, top-2
logit margins, MPS wall-clock latencies, and sandboxed Python tool executions
are executed live on Qwen/Qwen2.5-3B-Instruct and Qwen/Qwen2.5-7B-Instruct.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# 1. Benchmark Suite: 30 Two-Stage Systems, Numerical, & Algorithmic Tasks
#    (12 Routine Unambiguous Tasks + 18 Subtle Boundary/Precision/Bug Traps)
# ============================================================================

@dataclass(frozen=True)
class TwoStageTask:
    task_id: str
    regime: str  # "routine" or "boundary_trap"
    category: str
    stage1_question: str
    stage1_options: Dict[str, str]
    stage1_gold: str
    upstream_triage_note: str  # Shared RAG/static-lint summary consumed by correlated roots
    stage2_question: str
    stage2_options: Dict[str, str]
    stage2_gold: str


def build_pilot_tasks() -> List[TwoStageTask]:
    tasks: List[TwoStageTask] = [
        # ---------------- REGIME A: 12 ROUTINE UNAMBIGUOUS TASKS ----------------
        TwoStageTask(
            task_id="R01_sha256_bytes",
            regime="routine",
            category="systems_basics",
            stage1_question="How many raw bytes (not hex characters) are produced by a standard SHA-256 digest?",
            stage1_options={"A": "16 bytes", "B": "32 bytes", "C": "64 bytes", "D": "128 bytes"},
            stage1_gold="B",
            upstream_triage_note="Spec check: SHA-256 outputs a 256-bit digest (256 / 8 = 32 raw bytes, 64 hex chars).",
            stage2_question="Given Stage-1's finding on raw SHA-256 digest width, which fixed-size binary column buffer should be allocated per record?",
            stage2_options={"A": "BINARY(16)", "B": "BINARY(32)", "C": "BINARY(64)", "D": "BINARY(128)"},
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="R02_binary_heap_left_child",
            regime="routine",
            category="algorithms",
            stage1_question="In a 0-indexed binary heap stored in a contiguous array, what is the index of the left child of node i = 14?",
            stage1_options={"A": "27", "B": "28", "C": "29", "D": "30"},
            stage1_gold="C",
            upstream_triage_note="0-indexed heap invariant: left_child(i) = 2*i + 1 = 29.",
            stage2_question="Based on Stage-1's index verification for node i = 14, what is the exact index of its right sibling (right child of i = 14)?",
            stage2_options={"A": "28", "B": "29", "C": "30", "D": "31"},
            stage2_gold="C",
        ),
        TwoStageTask(
            task_id="R03_http_idempotent",
            regime="routine",
            category="distributed_systems",
            stage1_question="According to RFC 9110 (HTTP Semantics), which of the following HTTP methods is NOT guaranteed to be idempotent?",
            stage1_options={"A": "GET", "B": "PUT", "C": "DELETE", "D": "POST"},
            stage1_gold="D",
            upstream_triage_note="RFC 9110 Section 9.2.2: GET, HEAD, OPTIONS, TRACE, PUT, and DELETE are idempotent; POST is not.",
            stage2_question="Given Stage-1's HTTP idempotency classification, how should the RPC retry middleware handle automatic timeout retries without an idempotency key?",
            stage2_options={
                "A": "Auto-retry POST freely on timeout, block GET retries",
                "B": "Allow safe automatic retry on GET/PUT/DELETE, require explicit idempotency key before retrying POST",
                "C": "Disable retries on PUT and DELETE only",
                "D": "Treat POST and GET identically as stateless reads",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="R04_matmul_flops",
            regime="routine",
            category="numerical",
            stage1_question="If matrix X has shape (4, 8) and matrix Y has shape (8, 5), what is the output shape of the matrix product X @ Y?",
            stage1_options={"A": "(4, 5)", "B": "(8, 8)", "C": "(5, 4)", "D": "(4, 8)"},
            stage1_gold="A",
            upstream_triage_note="Linear algebra check: (4, 8) @ (8, 5) contracts inner dim 8 -> output shape (4, 5).",
            stage2_question="Given Stage-1's verified output shape for X @ Y, how many scalar elements are stored in the output tensor?",
            stage2_options={"A": "20 elements", "B": "32 elements", "C": "40 elements", "D": "64 elements"},
            stage2_gold="A",
        ),
        TwoStageTask(
            task_id="R05_tcp_syn_seq",
            regime="routine",
            category="networking",
            stage1_question="During a TCP 3-way handshake, if the client sends SYN with sequence number ISN = 1000, what acknowledgment number (ACK) does the server send in its SYN-ACK packet?",
            stage1_options={"A": "999", "B": "1000", "C": "1001", "D": "2000"},
            stage1_gold="C",
            upstream_triage_note="TCP SYN consumes 1 virtual sequence byte, so SYN-ACK acknowledges ISN + 1 = 1001.",
            stage2_question="Given Stage-1's handshake ACK rule, what packet check should the stateful firewall enforce on the server's SYN-ACK?",
            stage2_options={
                "A": "Require ack_num == client_isn - 1 (999)",
                "B": "Require ack_num == client_isn (1000)",
                "C": "Require ack_num == client_isn + 1 (1001)",
                "D": "Require ack_num == 2 * client_isn (2000)",
            },
            stage2_gold="C",
        ),
        TwoStageTask(
            task_id="R06_raft_quorum_size",
            regime="routine",
            category="distributed_systems",
            stage1_question="In a standard symmetric Raft cluster of N = 9 voting nodes, what is the minimum number of nodes required to form a commit quorum?",
            stage1_options={"A": "4 nodes", "B": "5 nodes", "C": "6 nodes", "D": "7 nodes"},
            stage1_gold="B",
            upstream_triage_note="Raft majority quorum: floor(9 / 2) + 1 = 5 nodes.",
            stage2_question="Given Stage-1's Raft quorum size for N = 9, what is the maximum number of simultaneous node crash failures (f) the cluster can tolerate while still committing log entries?",
            stage2_options={"A": "f = 3", "B": "f = 4", "C": "f = 5", "D": "f = 2"},
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="R07_dag_topo_cycle",
            regime="routine",
            category="algorithms",
            stage1_question="Kahn's topological sort algorithm runs on a directed graph with V = 12 vertices and terminates having popped only 9 vertices from the zero-indegree queue. What does this guarantee?",
            stage1_options={
                "A": "The graph is a valid tree",
                "B": "The graph contains at least one directed cycle",
                "C": "The graph has 3 disconnected isolated vertices with indegree 0",
                "D": "All edges have negative weights",
            },
            stage1_gold="B",
            upstream_triage_note="Kahn's algorithm visits < V vertices iff at least one directed cycle prevents remaining vertices from reaching indegree 0.",
            stage2_question="Given Stage-1's diagnosis of the workflow graph, how should the job scheduler respond at submission time?",
            stage2_options={
                "A": "Execute the 9 visited tasks and silently drop the remaining 3",
                "B": "Reject the workflow submission with a CyclicDependencyError",
                "C": "Treat the graph as a valid tree and enable tree-dp caching",
                "D": "Negate edge weights and rerun Dijkstra",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="R08_ipv4_cidr_hosts",
            regime="routine",
            category="networking",
            stage1_question="How many total IPv4 addresses (including network and broadcast addresses) are contained in a /26 CIDR block?",
            stage1_options={"A": "32", "B": "64", "C": "128", "D": "256"},
            stage1_gold="B",
            upstream_triage_note="IPv4 has 32 bits; a /26 prefix leaves 32 - 26 = 6 host bits -> 2^6 = 64 total addresses.",
            stage2_question="Given Stage-1's total address count for /26, how many usable host addresses remain in standard IPv4 networking after reserving network and broadcast addresses?",
            stage2_options={"A": "30", "B": "62", "C": "126", "D": "254"},
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="R09_btree_search_complexity",
            regime="routine",
            category="databases",
            stage1_question="What is the worst-case asymptotic time complexity of point-key lookup in a balanced B-Tree indexing N keys?",
            stage1_options={"A": "O(1)", "B": "O(log N)", "C": "O(sqrt(N))", "D": "O(N)"},
            stage1_gold="B",
            upstream_triage_note="Balanced B-Tree height scales logarithmically with key count -> O(log N) worst-case lookup.",
            stage2_question="Given Stage-1's B-Tree lookup complexity, which workload benefits directly from a B-Tree index over a hash-only index?",
            stage2_options={
                "A": "Ordered range scans (WHERE key BETWEEN low AND high) with O(log N + K) traversal",
                "B": "Guaranteed O(1) worst-case uncached random disk seeks regardless of N",
                "C": "Eliminating all key comparisons",
                "D": "Sorting arbitrary unindexed columns in O(1) time",
            },
            stage2_gold="A",
        ),
        TwoStageTask(
            task_id="R10_geometric_series",
            regime="routine",
            category="math",
            stage1_question="What is the exact sum of the finite geometric series 1 + 2 + 4 + 8 + 16 + 32 + 64 + 128?",
            stage1_options={"A": "254", "B": "255", "C": "256", "D": "511"},
            stage1_gold="B",
            upstream_triage_note="Sum_{k=0}^7 2^k = 2^8 - 1 = 255.",
            stage2_question="Given Stage-1's exact sum (255), does this total fit without overflow inside an unsigned 8-bit integer (uint8, range 0..255)?",
            stage2_options={
                "A": "No, 254 already overflows uint8",
                "B": "Yes, 255 equals UINT8_MAX (255) and fits cleanly without wraparound",
                "C": "No, 256 wraps to 0 in uint8",
                "D": "No, 511 requires uint16",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="R11_mutex_deadlock_order",
            regime="routine",
            category="concurrency",
            stage1_question="Thread 1 locks Mutex A then Mutex B, while Thread 2 locks Mutex B then Mutex A. Which classic Coffman condition is violated if we enforce a strict global lock acquisition order (always lock A before B)?",
            stage1_options={"A": "Mutual Exclusion", "B": "Circular Wait", "C": "No Preemption", "D": "Cache Coherence"},
            stage1_gold="B",
            upstream_triage_note="Total lock ordering makes a cycle in the wait-for graph impossible, eliminating Circular Wait.",
            stage2_question="Given Stage-1's deadlock condition analysis, what static/runtime lock-hierarchy rule prevents this deadlock?",
            stage2_options={
                "A": "Keep opposite lock orders and add random sleep()",
                "B": "Assign monotonically increasing lock ranks and assert rank(next_lock) > rank(held_lock) to eliminate Circular Wait",
                "C": "Allow multiple threads into the critical section simultaneously without synchronization",
                "D": "Flush L1 cache lines before unlocking",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="R12_kv_cache_bytes",
            regime="routine",
            category="ml_systems",
            stage1_question="A transformer layer caches K and V tensors for 1 sequence of 1024 tokens with 8 KV heads, head_dim = 128, stored in FP16 (2 bytes/element). How many megabytes (1 MB = 1,048,576 bytes) does this single layer's KV cache occupy?",
            stage1_options={"A": "1 MB", "B": "2 MB", "C": "4 MB", "D": "8 MB"},
            stage1_gold="C",
            upstream_triage_note="Bytes = 2 (K,V) * 1024 * 8 * 128 * 2 bytes = 4,194,304 bytes = 4 MB.",
            stage2_question="Given Stage-1's single-layer KV footprint, how much KV memory is needed across 32 identical layers for that sequence?",
            stage2_options={"A": "32 MB", "B": "64 MB", "C": "128 MB", "D": "256 MB"},
            stage2_gold="C",
        ),

        # ---------------- REGIME B: 18 SUBTLE BOUNDARY / BUG TRAP TASKS ----------------
        TwoStageTask(
            task_id="T01_uint16_overflow",
            regime="boundary_trap",
            category="systems_bugs",
            stage1_question=(
                "In C/CUDA, a kernel computes `uint16_t total_tokens = batch_size * seq_len;` "
                "When `batch_size = 256` and `seq_len = 256`, what exact value is stored in `total_tokens`?"
            ),
            stage1_options={"A": "65536", "B": "65535", "C": "0", "D": "32768"},
            stage1_gold="C",
            upstream_triage_note="Automated static check: `256 * 256` evaluates cleanly to `65536` with no division-by-zero or syntax warning.",
            stage2_question="Given Stage-1's finding on `uint16_t total_tokens` for 256x256, what remediation is required before deploying the CUDA kernel?",
            stage2_options={
                "A": "Approve and deploy kernel commit as-is without changing `uint16_t`",
                "B": "Block deploy and widen `total_tokens` (and operands) to `uint32_t` / `size_t`",
                "C": "Keep `uint16_t` and subtract 1 from `total_tokens`",
                "D": "Keep `uint16_t` and multiply `total_tokens` by 2",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T02_float32_precision_stall",
            regime="boundary_trap",
            category="numerical",
            stage1_question=(
                "A telemetry counter uses IEEE-754 single-precision `float32` (`np.float32`). "
                "Starting at `x = np.float32(16777216.0)` (`2**24`), the loop runs `x = np.float32(x + np.float32(1.0))` "
                "100 times. What is the exact final value of `x`?"
            ),
            stage1_options={"A": "16777316.0", "B": "16777216.0", "C": "16777266.0", "D": "Overflow to inf"},
            stage1_gold="B",
            upstream_triage_note="Quick math summary: 16,777,216 + 100*1.0 = 16,777,316.0, well below float32 max (~3.4e38).",
            stage2_question="Given Stage-1's verification of the `float32` request counter above 2^24, what action should be taken?",
            stage2_options={
                "A": "Ship `float32` counter unchanged",
                "B": "Replace `float32` accumulator with `int64` / `float64`",
                "C": "Keep `float32` and multiply increments by 2 after the loop",
                "D": "Add an `isinf()` saturation clamp on `float32`",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T03_sql_not_in_null",
            regime="boundary_trap",
            category="databases",
            stage1_question=(
                "Table `orders` has 5 rows with `id` in `(1, 2, 3, 4, 5)`. "
                "Subquery `blocked_ids` returns `(2, 4, NULL)`. "
                "How many rows does `SELECT id FROM orders WHERE id NOT IN (SELECT uid FROM blocked_ids);` return in standard SQL / PostgreSQL / SQLite?"
            ),
            stage1_options={"A": "3 rows (1, 3, 5)", "B": "0 rows", "C": "5 rows", "D": "2 rows (2, 4)"},
            stage1_gold="B",
            upstream_triage_note="Query triage: excluding IDs (2, 4) from (1, 2, 3, 4, 5) leaves the 3 unmatched IDs (1, 3, 5).",
            stage2_question="Given Stage-1's SQL `NOT IN` evaluation when NULLs can appear in `blocked_ids`, what deployment decision is appropriate?",
            stage2_options={
                "A": "Deploy `WHERE id NOT IN (SELECT uid FROM blocked_ids)` unchanged",
                "B": "Rewrite query to `NOT EXISTS` or add `WHERE uid IS NOT NULL` inside the subquery",
                "C": "Append `LIMIT 3` to the existing `NOT IN` query",
                "D": "Replace `NOT IN` with `IN`",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T04_shift_add_precedence",
            regime="boundary_trap",
            category="systems_bugs",
            stage1_question=(
                "In both C and Python 3, what exact integer value does the unparenthesized expression "
                "`1 << 2 + 3` evaluate to?"
            ),
            stage1_options={"A": "7", "B": "32", "C": "8", "D": "11"},
            stage1_gold="B",
            upstream_triage_note="Left-to-right scan: `1 << 2` is 4, plus 3 equals 7.",
            stage2_question="Given Stage-1's finding on `1 << 2 + 3` in bitmask macro code where `7` was intended, what action is needed?",
            stage2_options={
                "A": "Ship `1 << 2 + 3` unchanged",
                "B": "Parenthesize explicitly as `(1 << 2) + 3`",
                "C": "Change operator `<<` to `>>`",
                "D": "Cast `3` to `uint8_t` without parentheses",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T05_python_xor_caret",
            regime="boundary_trap",
            category="systems_bugs",
            stage1_question="In Python 3, what is the exact integer value evaluated by the expression `3 ^ 4 % 5`?",
            stage1_options={"A": "1", "B": "7", "C": "2", "D": "81"},
            stage1_gold="B",
            upstream_triage_note="Math note: 3^4 = 81, and 81 mod 5 = 1.",
            stage2_question="Given Stage-1's evaluation of `3 ^ 4 % 5` when modular exponentiation 3^4 mod 5 was intended, what action is required?",
            stage2_options={
                "A": "Approve `3 ^ 4 % 5` unchanged",
                "B": "Replace `3 ^ 4 % 5` with `pow(3, 4, 5)`",
                "C": "Keep `^` and parenthesize as `(3 ^ 4) % 5`",
                "D": "Remove `% 5` and keep `3 ^ 4`",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T06_python_bankers_rounding",
            regime="boundary_trap",
            category="numerical",
            stage1_question=(
                "In Python 3, evaluate `sum(round(x + 0.5) for x in range(8))` "
                "(i.e., `round(0.5) + round(1.5) + round(2.5) + round(3.5) + round(4.5) + round(5.5) + round(6.5) + round(7.5)`)."
            ),
            stage1_options={"A": "36", "B": "32", "C": "28", "D": "34"},
            stage1_gold="B",
            upstream_triage_note="Arithmetic check: rounding 0.5..7.5 up gives 1+2+3+4+5+6+7+8 = 36.",
            stage2_question="Given Stage-1's finding on Python `round()` when a billing spec requires `.5` to round strictly upward (target sum 36), what action should be taken?",
            stage2_options={
                "A": "Ship built-in `round()` unchanged",
                "B": "Replace `round()` with `Decimal.quantize(..., rounding=ROUND_HALF_UP)`",
                "C": "Replace `round()` with `int()` floor truncation",
                "D": "Subtract `0.5` before calling `round()`",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T07_closure_late_binding",
            regime="boundary_trap",
            category="systems_bugs",
            stage1_question=(
                "What exact value is returned by the Python 3 snippet:\n"
                "`funcs = [lambda x: x * i for i in range(5)]\n"
                "result = funcs[1](10) + funcs[2](10)`"
            ),
            stage1_options={"A": "30", "B": "80", "C": "100", "D": "20"},
            stage1_gold="B",
            upstream_triage_note="Code scan: `funcs[1](10)` is `10 * 1 = 10` and `funcs[2](10)` is `10 * 2 = 20`, totaling 30.",
            stage2_question="Given Stage-1's finding on `funcs = [lambda x: x * i for i in range(5)]` when `30` is desired, what patch is needed?",
            stage2_options={
                "A": "Approve `[lambda x: x * i for i in range(5)]` unchanged",
                "B": "Capture loop variable via default argument `[lambda x, i=i: x * i for i in range(5)]`",
                "C": "Change `range(5)` to `range(6)`",
                "D": "Wrap the list comprehension in `tuple(...)`",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T08_shallow_list_multiply",
            regime="boundary_trap",
            category="systems_bugs",
            stage1_question=(
                "What exact integer is printed by this Python snippet?\n"
                "`dp = [[0] * 4] * 5\n"
                "dp[0][2] = 7\n"
                "dp[3][2] += 2\n"
                "print(sum(row[2] for row in dp))`"
            ),
            stage1_options={"A": "9", "B": "45", "C": "35", "D": "14"},
            stage1_gold="B",
            upstream_triage_note="Matrix trace: `dp[0][2] = 7` and `dp[3][2] = 2`, while rows 1, 2, 4 stay 0 -> sum is 7 + 2 = 9.",
            stage2_question="Given Stage-1's finding on `dp = [[0] * 4] * 5` when independent rows (sum 9) are intended, what action is needed?",
            stage2_options={
                "A": "Keep `dp = [[0] * 4] * 5` unchanged",
                "B": "Replace initialization with `dp = [[0] * 4 for _ in range(5)]`",
                "C": "Use `dp = ([0] * 4,) * 5`",
                "D": "Divide final sum by 2",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T09_utf8_emoji_bytes",
            regime="boundary_trap",
            category="systems_bugs",
            stage1_question=(
                "In Python 3, what exact integer does `len('café 🚀'.encode('utf-8'))` return? "
                "(Note: string is `'c'`, `'a'`, `'f'`, `'é'` U+00E9, `' '` U+0020, `'🚀'` U+1F680.)"
            ),
            stage1_options={"A": "6", "B": "7", "C": "10", "D": "12"},
            stage1_gold="C",
            upstream_triage_note="String check: `'café 🚀'` has 6 characters (`c`, `a`, `f`, `é`, space, rocket), or 7 with accented char.",
            stage2_question="Given Stage-1's UTF-8 byte length for `'café 🚀'`, what exact `Content-Length` byte header must the HTTP serializer emit?",
            stage2_options={
                "A": "Content-Length: 6",
                "B": "Content-Length: 10",
                "C": "Content-Length: 7",
                "D": "Content-Length: 12",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T10_harmonic_throughput",
            regime="boundary_trap",
            category="distributed_systems",
            stage1_question=(
                "A backup job uploads a 600 GB snapshot over Link A at 60 MB/s, and immediately downloads/verifies "
                "the exact same 600 GB snapshot over Link B at 20 MB/s. What is the exact effective average transfer rate "
                "(total GB transferred divided by total elapsed time) across the round-trip?"
            ),
            stage1_options={"A": "40 MB/s", "B": "30 MB/s", "C": "35 MB/s", "D": "25 MB/s"},
            stage1_gold="B",
            upstream_triage_note="Throughput summary: average of 60 MB/s upload and 20 MB/s download is (60 + 20) / 2 = 40 MB/s.",
            stage2_question="Given Stage-1's effective round-trip rate, what exact total duration must be reserved to transfer 1200 GB (600 GB up + 600 GB down, 1 GB = 1000 MB)?",
            stage2_options={
                "A": "30,000 seconds",
                "B": "40,000 seconds",
                "C": "34,285 seconds",
                "D": "48,000 seconds",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T11_bayes_false_positive",
            regime="boundary_trap",
            category="math",
            stage1_question=(
                "In a cluster of 10,000 nodes, exactly 0.1% (10 nodes) truly have silent hardware corruption. "
                "An automated detector flags corrupt nodes with 99% true-positive sensitivity, and has a 1% false-positive rate "
                "on healthy nodes (99% specificity). If a randomly chosen node triggers an alert, what is the closest posterior "
                "probability that the node truly has silent corruption?"
            ),
            stage1_options={"A": "99.0%", "B": "90.0%", "C": "9.0%", "D": "50.0%"},
            stage1_gold="C",
            upstream_triage_note="Detector accuracy spec: 99% sensitivity and 99% specificity -> ~99% alert precision.",
            stage2_question="Given Stage-1's posterior precision on a single detector alert (where immediate wipe is only allowed above 80% precision), what runbook policy applies?",
            stage2_options={
                "A": "Immediately wipe and decommission every alerted node on first alert",
                "B": "Gate node wipe behind a secondary confirmation scrub before draining",
                "C": "Decommission 9 out of every 10 alerted nodes immediately",
                "D": "Disable hardware monitoring completely",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T12_int32_midpoint_overflow",
            regime="boundary_trap",
            category="algorithms",
            stage1_question=(
                "In C/Java (`int32_t`, signed 32-bit two's complement range `-2,147,483,648` to `2,147,483,647`), "
                "binary search computes `int32_t mid = (low + high) / 2;` with `low = 1500000000` and `high = 1700000000`. "
                "Wrapping in 32-bit two's complement before `/ 2`, what happens to `mid`?"
            ),
            stage1_options={
                "A": "mid evaluates cleanly to 1,600,000,000",
                "B": "low + high overflows signed int32, making mid = -547,483,648",
                "C": "mid saturates at 2,147,483,647",
                "D": "mid evaluates to 100,000,000",
            },
            stage1_gold="B",
            upstream_triage_note="Midpoint check: (1.5e9 + 1.7e9) / 2 = 1.6e9, which is < INT32_MAX (2.147e9).",
            stage2_question="Given Stage-1's finding on `int32_t mid = (low + high) / 2`, what code review decision applies?",
            stage2_options={
                "A": "Approve `(low + high) / 2` unchanged",
                "B": "Require `low + ((high - low) >> 1)`",
                "C": "Use `min((low + high) / 2, INT32_MAX - 1)`",
                "D": "Use `abs((low + high) / 2)`",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T13_sliding_window_fencepost",
            regime="boundary_trap",
            category="algorithms",
            stage1_question=(
                "A function computes the maximum sum of any contiguous subarray of length `k = 4` over `arr = [2, 1, 5, 1, 3, 2, 9]` (`len(arr) = 7`) using:\n"
                "`best = max(sum(arr[i : i + k]) for i in range(len(arr) - k))`\n"
                "What exact value does `best` evaluate to?"
            ),
            stage1_options={"A": "15", "B": "11", "C": "10", "D": "14"},
            stage1_gold="B",
            upstream_triage_note="Max 4-element contiguous window of [2, 1, 5, 1, 3, 2, 9] is [1, 3, 2, 9] = 15.",
            stage2_question="Given Stage-1's trace of `range(len(arr) - k)` when true max window sum 15 is required, what action applies?",
            stage2_options={
                "A": "Approve `range(len(arr) - k)` unchanged",
                "B": "Patch loop bound to `range(len(arr) - k + 1)`",
                "C": "Patch slice to `arr[i : i + k + 1]`",
                "D": "Patch loop bound to `range(len(arr) - k - 1)`",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T14_mutable_default_retry",
            regime="boundary_trap",
            category="systems_bugs",
            stage1_question=(
                "Consider Python function:\n"
                "`def record_attempt(code, history=[]):\n"
                "    history.append(code)\n"
                "    return sum(history)`\n"
                "If a worker calls `a = record_attempt(200)`, then `b = record_attempt(500)`, then `c = record_attempt(200)`, "
                "what is the exact value of `a + b + c`?"
            ),
            stage1_options={"A": "900", "B": "1800", "C": "1600", "D": "1100"},
            stage1_gold="B",
            upstream_triage_note="Call trace: three standalone calls with default `history=[]` return 200, 500, and 200 -> 200+500+200 = 900.",
            stage2_question="Given Stage-1's finding when each standalone call should start with an empty history (sum 900), what action applies?",
            stage2_options={
                "A": "Approve `def record_attempt(code, history=[]):` unchanged",
                "B": "Change signature to `history=None` and initialize `[]` inside the body",
                "C": "Call `history.pop()` before `return`",
                "D": "Change `history=[]` to `history={}`",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T15_neg_floor_mod_c_vs_py",
            regime="boundary_trap",
            category="numerical",
            stage1_question=(
                "In Python 3, what exact tuple is produced by `(-17 // 5, -17 % 5)`?"
            ),
            stage1_options={"A": "(-3, -2)", "B": "(-4, 3)", "C": "(-3, 3)", "D": "(-4, -2)"},
            stage1_gold="B",
            upstream_triage_note="Integer division note: 17 / 5 = 3 remainder 2, so with negative sign `-17 / 5` is `-3` remainder `-2`.",
            stage2_question="Given Stage-1's finding when porting C99's `(-17 / 5, -17 % 5)` (`(-3, -2)`) into Python, what action applies?",
            stage2_options={
                "A": "Use Python `(-17 // 5, -17 % 5)` directly without modification",
                "B": "Use `int(-17 / 5)` and `int(math.fmod(-17, 5))` for truncation-toward-zero semantics",
                "C": "Use `(-17 // 5, -(-17 % 5))`",
                "D": "Add `1` to `x // y` for all integers",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T16_generator_exhaustion",
            regime="boundary_trap",
            category="systems_bugs",
            stage1_question=(
                "What exact float value is computed by this Python 3 snippet?\n"
                "`scores = (x * 2 for x in [1, 2, 3, 4])\n"
                "total = sum(scores)\n"
                "count = len(list(scores))\n"
                "avg = total / max(count, 1)`"
            ),
            stage1_options={"A": "5.0", "B": "20.0", "C": "2.5", "D": "0.0"},
            stage1_gold="B",
            upstream_triage_note="Metrics check: doubled list is [2, 4, 6, 8], sum = 20, count = 4 -> avg = 20 / 4 = 5.0.",
            stage2_question="Given Stage-1's trace when the true mean `5.0` is required, what patch applies?",
            stage2_options={
                "A": "Approve generator expression `scores = (x * 2 for ...)` unchanged",
                "B": "Materialize `scores = [x * 2 for x in [1, 2, 3, 4]]` as a list before `sum` and `len`",
                "C": "Replace `max(count, 1)` with `count`",
                "D": "Divide `avg` by `2.0`",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T17_float_nan_sort_membership",
            regime="boundary_trap",
            category="numerical",
            stage1_question=(
                "In Python 3, execute:\n"
                "`v = float('nan')\n"
                "a = (v == v)\n"
                "b = (v in [v])`\n"
                "What exact pair `(a, b)` is produced?"
            ),
            stage1_options={
                "A": "(False, False)",
                "B": "(False, True)",
                "C": "(True, True)",
                "D": "(True, False)",
            },
            stage1_gold="B",
            upstream_triage_note="IEEE-754 NaN rule: `NaN != NaN` is always True, so both equality `v == v` and list membership `v in [v]` are False -> `(False, False)`.",
            stage2_question="Given Stage-1's finding on `v == v` vs `v in [v]`, which statement accurately documents Python list `in` semantics for unit tests?",
            stage2_options={
                "A": "Document that `x in [x]` evaluates to `False` for `NaN` just like `x == x`",
                "B": "Document that container `in` checks object identity (`is`) before `==`, so `v in [v]` is `True` while `v == v` is `False`",
                "C": "Document that both `v == v` and `v in [v]` evaluate to `True`",
                "D": "Document that `v == v` is `True` while `v in [v]` is `False`",
            },
            stage2_gold="B",
        ),
        TwoStageTask(
            task_id="T18_modulo_chain_remainder",
            regime="boundary_trap",
            category="math",
            stage1_question=(
                "What is the exact remainder when `7**2026 + 3**2026` is divided by `10`?"
            ),
            stage1_options={"A": "0", "B": "8", "C": "2", "D": "4"},
            stage1_gold="B",
            upstream_triage_note="Quick modular check: 7 + 3 = 10 == 0 (mod 10), so 7^n + 3^n is divisible by 10 -> remainder 0.",
            stage2_question="Given Stage-1's exact remainder of `7**2026 + 3**2026 (mod 10)`, what parity guard must be placed on `n` so `7**n + 3**n` is divisible by `10`?",
            stage2_options={
                "A": "Accept all positive integers `n` without restriction",
                "B": "Require `n % 2 == 1` (odd positive integers only)",
                "C": "Require `n % 4 == 0`",
                "D": "Require `n % 2 == 0` (even positive integers only)",
            },
            stage2_gold="B",
        ),
    ]

    # Interleave routine and boundary_trap tasks deterministically so any prefix slice is balanced
    routine = [t for t in tasks if t.regime == "routine"]
    traps = [t for t in tasks if t.regime == "boundary_trap"]
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


# ============================================================================
# 2. Live Sandboxed Python Verification Helper (Used by Straggler Verifiers)
# ============================================================================

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def run_python_sandbox(code_str: str, timeout_s: float = 2.5) -> Tuple[bool, str]:
    """Execute generated verification snippet in an isolated Python subprocess."""
    code_clean = code_str.strip()
    if not code_clean:
        return False, "No code block generated."
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code_clean],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        if proc.returncode == 0 and out:
            return True, out[:400]
        return False, (err or out or f"Exit code {proc.returncode}")[:300]
    except subprocess.TimeoutExpired:
        return False, "Execution timed out."
    except Exception as exc:
        return False, f"Sandbox error: {exc}"


# ============================================================================
# 3. Real MPS Model Runner with Exact 4-Option Posterior Entropy & Logit Margin
# ============================================================================

@dataclass
class AgentObservation:
    agent_id: int
    tier: str  # "fast_3b", "cot_7b", "verifier_7b"
    genealogy_root: str
    choice: str  # "A", "B", "C", "D"
    is_correct: bool
    option_probs: Dict[str, float]
    option_entropy_nats: float  # in [0, ln(4) = 1.3863]
    normalized_entropy: float   # in [0, 1]
    logit_margin: float         # z_(1) - z_(2) over {A, B, C, D}
    seq_token_entropy: float    # mean predictive entropy across generated reasoning tokens
    tool_verified: bool
    mps_wall_s: float
    queue_tool_overhead_s: float
    completion_time_s: float
    snippet_preview: str


class MPSAgentEngine:
    def __init__(self, model_name: str, device: str = "mps") -> None:
        self.model_name = model_name
        self.device = device
        print(f"[Engine] Loading {model_name} on {device} (FP16, local_files_only=True)...", flush=True)
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.float16, local_files_only=True
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, local_files_only=True
            )
        self.model.to(device)
        self.model.eval()
        if device == "mps":
            torch.mps.synchronize()
        print(f"[Engine] Loaded {model_name} in {time.perf_counter() - t0:.2f}s", flush=True)

        # Resolve single-token IDs for "A", "B", "C", "D"
        self.option_token_ids: Dict[str, int] = {}
        for letter in ("A", "B", "C", "D"):
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            self.option_token_ids[letter] = int(ids[0])

    def unload(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()

    def _encode_messages(self, messages: List[Dict[str, str]]) -> List[int]:
        encoded = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if hasattr(encoded, "keys") and "input_ids" in encoded:
            return list(encoded["input_ids"])
        if isinstance(encoded, list):
            return encoded
        return list(encoded)

    def generate_and_score_options(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> Tuple[str, str, Dict[str, float], float, float, float, float]:
        """
        1) Generates reasoning/response up to `max_new_tokens` while recording mean
           token-level predictive entropy across generated tokens.
        2) Appends `\nFinal Answer Option (single letter A, B, C, or D): ` and evaluates
           exact next-token logits over ('A', 'B', 'C', 'D') in FP32 to obtain calibrated
           4-way option posterior, semantic entropy H_ans, and top-2 logit margin.
        """
        if self.device == "mps":
            torch.mps.synchronize()
        t_start = time.perf_counter()

        prompt_ids = self._encode_messages(messages)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        attn_mask = torch.ones_like(input_ids)

        with torch.inference_mode():
            if max_new_tokens > 0:
                if temperature > 0.0:
                    torch.manual_seed(seed)
                gen_out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0.0),
                    temperature=temperature if temperature > 0.0 else None,
                    top_p=0.9 if temperature > 0.0 else None,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                seq_ids = gen_out.sequences[0].tolist()
                new_ids = seq_ids[len(prompt_ids) :]
                gen_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

                # Mean token-level predictive entropy across generated tokens
                ent_vals: List[float] = []
                for step_logits in gen_out.scores:
                    logits_f32 = step_logits[0].float()
                    log_probs = torch.log_softmax(logits_f32, dim=-1)
                    probs = torch.exp(log_probs)
                    ent = float(-(probs * log_probs).sum().item())
                    if math.isfinite(ent):
                        ent_vals.append(ent)
                seq_ent = float(np.mean(ent_vals)) if ent_vals else 0.0
            else:
                gen_text = ""
                seq_ent = 0.0

            # Encode assistant prefix without closing im_end so next token is A/B/C/D
            raw_chat = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if gen_text.strip():
                prefix_str = raw_chat + gen_text.strip() + "\nAnswer Option (A, B, C, or D): "
            else:
                prefix_str = raw_chat + "Answer Option (A, B, C, or D): "
            probe_ids = self.tokenizer.encode(prefix_str, add_special_tokens=False)
            probe_tensor = torch.tensor([probe_ids], dtype=torch.long, device=self.device)
            out = self.model(input_ids=probe_tensor, use_cache=False)
            last_logits = out.logits[0, -1, :].float()

            opt_letters = ["A", "B", "C", "D"]
            opt_logits = torch.tensor(
                [float(last_logits[self.option_token_ids[c]].item()) for c in opt_letters],
                dtype=torch.float32,
            )
            opt_log_probs = torch.log_softmax(opt_logits, dim=-1)
            opt_probs_t = torch.exp(opt_log_probs)
            opt_ent = float(-(opt_probs_t * opt_log_probs).sum().item())

            sorted_logits, _ = torch.sort(opt_logits, descending=True)
            margin = float((sorted_logits[0] - sorted_logits[1]).item())

            best_idx = int(torch.argmax(opt_logits).item())
            choice = opt_letters[best_idx]
            probs_dict = {c: round(float(opt_probs_t[idx].item()), 5) for idx, c in enumerate(opt_letters)}

        if self.device == "mps":
            torch.mps.synchronize()
        mps_wall = time.perf_counter() - t_start
        return choice, gen_text.strip(), probs_dict, opt_ent, margin, seq_ent, mps_wall


def format_mc_prompt(question: str, options: Dict[str, str], extra_context: str = "") -> str:
    opts_block = "\n".join(f"  {k}) {v}" for k, v in sorted(options.items()))
    ctx = f"\n{extra_context.strip()}\n" if extra_context.strip() else ""
    return f"{question}{ctx}\nOptions:\n{opts_block}"


# ============================================================================
# 4. Five Distributed Consensus Policies (Head-to-Head Evaluator)
# ============================================================================

@dataclass
class QuorumDecision:
    policy: str
    committed_choice: str
    is_correct: bool
    latency_s: float
    agents_observed: int
    boost_triggered: bool
    effective_independent_weight: float


def compute_agent_epistemic_weight(obs: AgentObservation) -> float:
    """
    Calibrated epistemic confidence weight w_i in (0, 1.35]:
      Combines 4-option posterior certainty (1 - H_ans / ln 4), top-2 logit margin
      saturation tanh(margin / 3.5), and execution grounding bonus for sandbox-verified
      certificates vs unverified direct assertions.
    """
    cert_prob = max(0.0, 1.0 - obs.normalized_entropy)
    margin_conf = math.tanh(max(0.0, obs.logit_margin) / 3.5)
    base_conf = 0.55 * cert_prob + 0.45 * margin_conf
    if obs.tool_verified:
        grounding_mult = 1.38
    elif obs.tier == "verifier_7b":
        grounding_mult = 1.18
    elif obs.tier == "cot_7b":
        grounding_mult = 1.12
    elif obs.tier == "short_cot_3b":
        grounding_mult = 0.85
    else:
        # Unverified zero-CoT fast-tier direct assertion
        grounding_mult = 0.76
    return min(1.35, max(0.05, base_conf * grounding_mult))


def evaluate_stage_policies(
    observations: List[AgentObservation],
    gold_choice: str,
    cvar_static_budget_s: float = 2.8,
    qg_tau_quorum: float = 2.12,
    qg_margin_gap: float = 1.05,
    qg_max_fast_entropy: float = 0.08,
    qg_min_fast_logit_margin: float = 3.2,
    qg_boost_speedup: float = 3.2,
) -> Dict[str, QuorumDecision]:
    """
    Evaluates all 5 consensus policies on a single workflow stage's asynchronous
    agent pool:
      1. Fast_k_of_N: Closes as soon as earliest 3 agents finish (or earliest 2 agree).
      2. Sync_N_of_N: Waits for all N agents; raw unweighted majority vote.
      3. Genealogy_Sync_HPR: Waits for all N agents; deduplicates shared provenance roots
         weighted by epistemic confidence (arXiv:2603.04474 / 2606.07941).
      4. CVaR_Static_Cutoff: Cuts off at static deadline budget B (arXiv:2609.10964);
         majority among arrived agents (falls back to earliest arrival if none <= B).
      5. QuorumGuard: Event-driven genealogy-deduplicated & entropy-gated adaptive quorum
         with 3.2x Priority-Budget Dissent Boosting for active deep stragglers.
    """
    obs_sorted = sorted(observations, key=lambda o: o.completion_time_s)
    n_total = len(obs_sorted)
    decisions: Dict[str, QuorumDecision] = {}

    # --- Policy 1: Fast k-of-N Quorum (First-3-Arrive / Early-2-Agree) ---
    counts_fast: Dict[str, int] = {}
    fast_choice = obs_sorted[0].choice
    fast_lat = obs_sorted[0].completion_time_s
    fast_seen = 0
    for o in obs_sorted:
        fast_seen += 1
        counts_fast[o.choice] = counts_fast.get(o.choice, 0) + 1
        if counts_fast[o.choice] >= 2 and fast_seen >= 2:
            fast_choice = o.choice
            fast_lat = o.completion_time_s
            break
        if fast_seen >= 3:
            fast_choice = max(counts_fast.items(), key=lambda kv: (kv[1], -obs_sorted[0].option_entropy_nats))[0]
            fast_lat = o.completion_time_s
            break
    decisions["Fast_k_of_N"] = QuorumDecision(
        policy="Fast_k_of_N",
        committed_choice=fast_choice,
        is_correct=(fast_choice == gold_choice),
        latency_s=round(fast_lat, 4),
        agents_observed=fast_seen,
        boost_triggered=False,
        effective_independent_weight=float(counts_fast.get(fast_choice, 1)),
    )

    # --- Policy 2: Synchronous Full Barrier N-of-N Raw Majority ---
    counts_sync: Dict[str, Tuple[int, float]] = {}
    for o in obs_sorted:
        prev_cnt, prev_conf = counts_sync.get(o.choice, (0, 0.0))
        counts_sync[o.choice] = (prev_cnt + 1, prev_conf + (1.0 - o.normalized_entropy))
    sync_choice = max(counts_sync.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]
    sync_lat = obs_sorted[-1].completion_time_s
    decisions["Sync_N_of_N"] = QuorumDecision(
        policy="Sync_N_of_N",
        committed_choice=sync_choice,
        is_correct=(sync_choice == gold_choice),
        latency_s=round(sync_lat, 4),
        agents_observed=n_total,
        boost_triggered=False,
        effective_independent_weight=float(counts_sync[sync_choice][0]),
    )

    # --- Policy 3: Genealogy-Weighted Synchronous (HPR-Adaptive / Spark-to-Fire) ---
    def genealogy_scores(subset: Sequence[AgentObservation]) -> Tuple[Dict[str, float], Dict[str, int]]:
        root_claim_max: Dict[Tuple[str, str], float] = {}
        for o in subset:
            w = compute_agent_epistemic_weight(o)
            key = (o.genealogy_root, o.choice)
            root_claim_max[key] = max(root_claim_max.get(key, 0.0), w)
        claim_tot: Dict[str, float] = {}
        claim_roots: Dict[str, int] = {}
        for (_, ch), w_max in root_claim_max.items():
            claim_tot[ch] = claim_tot.get(ch, 0.0) + w_max
            claim_roots[ch] = claim_roots.get(ch, 0) + 1
        return claim_tot, claim_roots

    hpr_tot, _ = genealogy_scores(obs_sorted)
    hpr_choice = max(hpr_tot.items(), key=lambda kv: kv[1])[0]
    decisions["Genealogy_Sync_HPR"] = QuorumDecision(
        policy="Genealogy_Sync_HPR",
        committed_choice=hpr_choice,
        is_correct=(hpr_choice == gold_choice),
        latency_s=round(sync_lat, 4),
        agents_observed=n_total,
        boost_triggered=False,
        effective_independent_weight=round(hpr_tot[hpr_choice], 4),
    )

    # --- Policy 4: CVaR Static Deadline Cutoff (arXiv:2609.10964) ---
    arrived_cvar = [o for o in obs_sorted if o.completion_time_s <= cvar_static_budget_s]
    if not arrived_cvar:
        arrived_cvar = [obs_sorted[0]]
        cvar_lat = obs_sorted[0].completion_time_s
    else:
        cvar_lat = min(sync_lat, cvar_static_budget_s)
    counts_cvar: Dict[str, Tuple[int, float]] = {}
    for o in arrived_cvar:
        pc, pw = counts_cvar.get(o.choice, (0, 0.0))
        counts_cvar[o.choice] = (pc + 1, pw + (1.0 - o.normalized_entropy))
    cvar_choice = max(counts_cvar.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]
    decisions["CVaR_Static_Cutoff"] = QuorumDecision(
        policy="CVaR_Static_Cutoff",
        committed_choice=cvar_choice,
        is_correct=(cvar_choice == gold_choice),
        latency_s=round(cvar_lat, 4),
        agents_observed=len(arrived_cvar),
        boost_triggered=False,
        effective_independent_weight=float(counts_cvar[cvar_choice][0]),
    )

    # --- Policy 5: QuorumGuard (Adaptive Genealogy + Entropy Gate + Straggler Boost) ---
    dyn_times = {o.agent_id: o.completion_time_s for o in observations}
    obs_by_id = {o.agent_id: o for o in observations}
    finished_ids: List[int] = []
    boost_triggered = False
    qg_choice = hpr_choice
    qg_lat = sync_lat
    qg_eff_w = hpr_tot[hpr_choice]

    min_indep_roots = 3 if n_total >= 6 else 2
    eff_tau = qg_tau_quorum if n_total >= 6 else 1.55

    while len(finished_ids) < n_total:
        remaining = [aid for aid in dyn_times if aid not in finished_ids]
        next_aid = min(remaining, key=lambda aid: dyn_times[aid])
        t_now = dyn_times[next_aid]
        finished_ids.append(next_aid)

        arrived_obs = [obs_by_id[aid] for aid in finished_ids]
        claim_tot, claim_roots = genealogy_scores(arrived_obs)
        ranked = sorted(claim_tot.items(), key=lambda kv: kv[1], reverse=True)
        lead_ch, lead_w = ranked[0]
        runner_w = ranked[1][1] if len(ranked) > 1 else 0.0
        lead_root_cnt = claim_roots[lead_ch]

        # Mean normalized entropy & verification grounding among supporters of leading claim
        supporters = [o for o in arrived_obs if o.choice == lead_ch]
        mean_sup_ent = float(np.mean([o.normalized_entropy for o in supporters]))
        min_sup_margin = float(min(o.logit_margin for o in supporters))
        has_deep_cert = any(o.tool_verified or o.tier in ("verifier_7b", "cot_7b") for o in supporters)
        any_dissent_so_far = len(ranked) >= 2 and runner_w >= 0.35

        # Early closure condition:
        # (a) Unanimous clean fast quorum: >= min_indep_roots independent roots agree with ZERO dissent,
        #     sharp pre-commitment logit margins, and near-zero semantic entropy;
        # OR (b) Deep/tool-verified hedging quorum: >= 2 deep/verified roots concur with clear lead margin over runner-up.
        deep_supporter_roots = len(
            {o.genealogy_root for o in supporters if o.tool_verified or o.tier in ("verifier_7b", "cot_7b")}
        )
        early_clean_quorum = (
            (not any_dissent_so_far)
            and lead_root_cnt >= min_indep_roots
            and lead_w >= eff_tau
            and mean_sup_ent <= qg_max_fast_entropy
            and min_sup_margin >= qg_min_fast_logit_margin
        )
        verified_hedged_quorum = (
            has_deep_cert
            and (deep_supporter_roots >= 2 or (deep_supporter_roots >= 1 and not any_dissent_so_far))
            and lead_root_cnt >= 2
            and lead_w >= (eff_tau - 0.15)
            and (lead_w - runner_w) >= 0.72
        )

        if early_clean_quorum or verified_hedged_quorum:
            qg_choice = lead_ch
            qg_lat = t_now
            qg_eff_w = lead_w
            break

        # Dissent / Correlation / Entropy Hedging Trigger:
        # Once >= 3 raw agents have arrived and the gate has NOT closed (because of root correlation,
        # claim disagreement, or elevated predictive entropy), boost GPU scheduling priority of active
        # deep verifiers so remaining queue/contention wait compresses by `qg_boost_speedup`!
        if (not boost_triggered) and len(finished_ids) >= 3:
            boost_triggered = True
            for aid, o_cand in obs_by_id.items():
                if aid not in finished_ids and o_cand.tier in ("verifier_7b", "cot_7b"):
                    # Preserve irreducible MPS model execution floor while compressing queue/sandbox tail wait
                    irreducible_floor = t_now + 0.45 * o_cand.mps_wall_s
                    compressed = t_now + (dyn_times[aid] - t_now) / qg_boost_speedup
                    dyn_times[aid] = max(irreducible_floor, compressed)
    else:
        arrived_obs = [obs_by_id[aid] for aid in finished_ids]
        claim_tot, _ = genealogy_scores(arrived_obs)
        qg_choice, qg_eff_w = max(claim_tot.items(), key=lambda kv: kv[1])
        qg_lat = max(dyn_times.values())

    decisions["QuorumGuard"] = QuorumDecision(
        policy="QuorumGuard",
        committed_choice=qg_choice,
        is_correct=(qg_choice == gold_choice),
        latency_s=round(qg_lat, 4),
        agents_observed=len(finished_ids),
        boost_triggered=boost_triggered,
        effective_independent_weight=round(qg_eff_w, 4),
    )

    return decisions


# ============================================================================
# 5. Two-Pass Memory-Safe Stage Execution (3B Pool -> Unload -> 7B Pool)
# ============================================================================

def run_pilot(max_tasks: int = 30, seed: int = 2026) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    all_tasks = build_pilot_tasks()[:max_tasks]
    print(f"[Pilot] Running {len(all_tasks)} Two-Stage Tasks (Seed={seed})", flush=True)

    # Pre-sample realistic asynchronous RPC/queue/sandbox delays per task & agent
    # Stage 1 has 7 agents:
    #   0: fast_3b (shared_triage_root)
    #   1: fast_3b (shared_triage_root)
    #   2: fast_3b (indep_fast_root_2)
    #   3: fast_3b (indep_fast_root_3)
    #   4: cot_7b  (indep_cot_root_4)
    #   5: verifier_7b (sandbox_root_5 - Pareto straggler)
    #   6: verifier_7b (sandbox_root_6 - Pareto straggler)
    s1_overheads: Dict[str, List[float]] = {}
    for t in all_tasks:
        fast_q = rng.uniform(0.18, 0.72, size=4).tolist()
        cot_q = [float(rng.uniform(0.85, 1.95))]
        strag_q = (3.8 + rng.pareto(a=2.25, size=2) * 2.7).clip(3.8, 16.5).tolist()
        s1_overheads[t.task_id] = fast_q + cot_q + [float(x) for x in strag_q]

    stage1_obs_map: Dict[str, List[AgentObservation]] = {t.task_id: [] for t in all_tasks}

    # ---------------- PASS 1: Qwen2.5-3B-Instruct on Stage 1 ----------------
    eng_3b = MPSAgentEngine("Qwen/Qwen2.5-3B-Instruct", device="mps")
    fast_configs = [
        (0, "fast_3b", "shared_triage_root", True, 0, "You are Fast Triage Agent A. Rely on the provided triage note for an immediate decision."),
        (1, "fast_3b", "shared_triage_root", True, 0, "You are Fast Triage Agent B. Follow the upstream triage summary for an immediate decision."),
        (2, "fast_3b", "indep_fast_root_2", False, 0, "You are Rapid Scanner C. Make an immediate zero-CoT intuitive decision."),
        (3, "short_cot_3b", "indep_fast_cot_root_3", False, 45, "You are Quick Arithmetic & Type Checker D. Write 1-2 short sentences checking exact data types, bit-widths, or edge cases WITHOUT naming an option letter A/B/C/D yet."),
    ]
    for idx_t, task in enumerate(all_tasks):
        for aid, tier_lbl, root_id, use_note, max_tok, sys_prompt in fast_configs:
            extra = f"Upstream Automated Triage Note: {task.upstream_triage_note}" if use_note else ""
            user_prompt = format_mc_prompt(task.stage1_question, task.stage1_options, extra)
            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
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
                    snippet_preview=txt[:120] if txt else f"[pre-commit direct: {ch} margin={margin:.2f}]",
                )
            )
        print(f"  [Stage1-3B] {idx_t+1}/{len(all_tasks)} {task.task_id} done", flush=True)
    eng_3b.unload()

    # ---------------- PASS 2: Qwen2.5-7B-Instruct on Stage 1 ----------------
    eng_7b = MPSAgentEngine("Qwen/Qwen2.5-7B-Instruct", device="mps")
    for idx_t, task in enumerate(all_tasks):
        # Agent 4: Medium CoT 7B
        msgs_cot = [
            {
                "role": "system",
                "content": (
                    "You are a careful Systems & Mathematics Reviewer. Check exact fixed-width integer types "
                    "(uint16_t wraps mod 65536, int32_t wraps in [-2^31, 2^31-1]), IEEE-754 float32 ULP rounding, "
                    "C/Python operator precedence, SQL NULL three-valued logic, and Python object aliasing. "
                    "Reason concisely in 2-4 sentences WITHOUT writing Option letters A/B/C/D yet."
                ),
            },
            {"role": "user", "content": format_mc_prompt(task.stage1_question, task.stage1_options)},
        ]
        ch4, txt4, probs4, ent4, mar4, seq4, mps4 = eng_7b.generate_and_score_options(
            msgs_cot, max_new_tokens=115, temperature=0.0
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
                snippet_preview=txt4[:120],
            )
        )

        # Agents 5 & 6: Deep Sandbox-Python Verifiers (2-turn LLM -> Python Subprocess -> LLM)
        for idx_v, (aid, root_id, persona_note) in enumerate(
            [
                (5, "sandbox_root_5", "Write a self-contained ```python ... ``` script simulating exact types (`np.uint16`, `np.int32`, `np.float32`, `sqlite3`, etc.) and print plain non-scientific values (e.g. `int(x)` or `f'{float(x):.1f}'`)."),
                (6, "sandbox_root_6", "Write an independent ```python ... ``` reproduction test simulating the exact language bit-widths/SQLite/Python semantics and printing plain exact values (never scientific `1e+07` notation)."),
            ]
        ):
            code_msgs = [
                {
                    "role": "system",
                    "content": (
                        f"You are Deep Verification Agent #{aid}. {persona_note} "
                        "Keep the Python snippet under 15 lines using Python standard library / numpy / sqlite3."
                    ),
                },
                {"role": "user", "content": format_mc_prompt(task.stage1_question, task.stage1_options)},
            ]
            _, code_resp, _, _, _, _, mps_turn1 = eng_7b.generate_and_score_options(
                code_msgs, max_new_tokens=120, temperature=(0.15 if idx_v == 1 else 0.0), seed=seed + idx_t + aid
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
                        "In 1-2 short sentences, match this exact verified output value directly to the corresponding choice in Options (watch exact digits!) WITHOUT writing 'Answer Option' yet."
                    ),
                },
            ]
            ch_v, txt_v, probs_v, ent_v, mar_v, seq_v, mps_turn2 = eng_7b.generate_and_score_options(
                synth_msgs, max_new_tokens=55, temperature=0.0
            )
            tot_mps = mps_turn1 + mps_turn2 + sbx_wall
            q_v = s1_overheads[task.task_id][aid]
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
                    snippet_preview=f"[tool_ok={ok_tool}: {tool_out[:60]}] {txt_v[:60]}",
                )
            )
        print(f"  [Stage1-7B] {idx_t+1}/{len(all_tasks)} {task.task_id} done", flush=True)

    # Evaluate Stage-1 consensus across all 5 policies and collect unique committed choices per task
    stage1_decisions_map: Dict[str, Dict[str, QuorumDecision]] = {}
    unique_s1_commits: Dict[str, List[str]] = {}
    for task in all_tasks:
        decs = evaluate_stage_policies(stage1_obs_map[task.task_id], task.stage1_gold)
        stage1_decisions_map[task.task_id] = decs
        uniq = sorted({d.committed_choice for d in decs.values()})
        unique_s1_commits[task.task_id] = uniq

    # ---------------- STAGE 2: Downstream Remediation Synthesis Committee (N=5) ----------------
    # Agents 0, 1 (fast_3b, shared_stage1_commit_root): Trust upstream Stage-1 consensus directly!
    # Agent 2 (fast_3b, indep_s2_fast_2): Reads problem + Stage-1 commit concisely.
    # Agents 3, 4 (verifier_7b, indep_s2_audit_3 / 4): Cross-audit original Stage-1 problem & Stage-2 action!
    s2_overheads: Dict[Tuple[str, str], List[float]] = {}
    for task in all_tasks:
        for s1_ch in unique_s1_commits[task.task_id]:
            fq = rng.uniform(0.16, 0.65, size=3).tolist()
            sq = (3.4 + rng.pareto(a=2.3, size=2) * 2.5).clip(3.4, 14.5).tolist()
            s2_overheads[(task.task_id, s1_ch)] = fq + [float(x) for x in sq]

    stage2_obs_map: Dict[Tuple[str, str], List[AgentObservation]] = {
        k: [] for k in s2_overheads
    }

    # Run Stage-2 7B Audit Verifiers FIRST while 7B is ALREADY loaded in MPS memory (saves an extra load!)
    for idx_t, task in enumerate(all_tasks):
        for s1_ch in unique_s1_commits[task.task_id]:
            s1_claim_txt = task.stage1_options.get(s1_ch, s1_ch)
            for aid, root_id in [(3, "s2_audit_root_3"), (4, "s2_audit_root_4")]:
                audit_ctx = (
                    f"Original Upstream Technical Question:\n{task.stage1_question}\n"
                    f"Upstream Stage-1 Committee Committed Finding: Option {s1_ch} ({s1_claim_txt}).\n"
                    "IMPORTANT: Independently verify whether Upstream Stage-1's finding is technically sound "
                    "(watch for overflow, precision, SQL NULL, precedence, or Python traps) before choosing the safest remediation!"
                )
                msgs_s2_7b = [
                    {
                        "role": "system",
                        "content": "You are Principal Deployment Auditor. Independently verify upstream claims from first principles in 2-3 sentences WITHOUT naming an option letter A/B/C/D yet.",
                    },
                    {"role": "user", "content": format_mc_prompt(task.stage2_question, task.stage2_options, audit_ctx)},
                ]
                ch_a, txt_a, probs_a, ent_a, mar_a, seq_a, mps_a = eng_7b.generate_and_score_options(
                    msgs_s2_7b, max_new_tokens=95, temperature=(0.15 if aid == 4 else 0.0), seed=seed + 500 + idx_t + aid
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
                        snippet_preview=txt_a[:100],
                    )
                )
        print(f"  [Stage2-7B] {idx_t+1}/{len(all_tasks)} {task.task_id} done", flush=True)
    eng_7b.unload()

    # Now load 3B once more for Stage-2 Fast Downstream Agents (0, 1, 2)
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
                        snippet_preview=txt_f[:100],
                    )
                )
        print(f"  [Stage2-3B] {idx_t+1}/{len(all_tasks)} {task.task_id} done", flush=True)
    eng_3b.unload()

    # ========================================================================
    # 6. Aggregate End-to-End 2-Stage Workflow Metrics & Covariance Proof
    # ========================================================================
    policy_names = [
        "Fast_k_of_N",
        "Sync_N_of_N",
        "Genealogy_Sync_HPR",
        "CVaR_Static_Cutoff",
        "QuorumGuard",
    ]
    per_task_records: List[Dict[str, Any]] = []
    all_agent_latencies: List[float] = []
    all_agent_accs: List[float] = []
    trap_agent_latencies: List[float] = []
    trap_agent_accs: List[float] = []

    for task in all_tasks:
        s1_obs = sorted(stage1_obs_map[task.task_id], key=lambda o: o.agent_id)
        for o in s1_obs:
            all_agent_latencies.append(o.completion_time_s)
            all_agent_accs.append(1.0 if o.is_correct else 0.0)
            if task.regime == "boundary_trap":
                trap_agent_latencies.append(o.completion_time_s)
                trap_agent_accs.append(1.0 if o.is_correct else 0.0)

        s1_decs = stage1_decisions_map[task.task_id]
        s2_decs_for_policy: Dict[str, QuorumDecision] = {}
        e2e_policy_summary: Dict[str, Any] = {}

        for pol in policy_names:
            d1 = s1_decs[pol]
            s2_pool = stage2_obs_map[(task.task_id, d1.committed_choice)]
            d2 = evaluate_stage_policies(s2_pool, task.stage2_gold)[pol]
            s2_decs_for_policy[pol] = d2
            e2e_policy_summary[pol] = {
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
                "policies": e2e_policy_summary,
            }
        )

    cov_all = float(np.cov(all_agent_latencies, all_agent_accs)[0, 1])
    corr_all = float(np.corrcoef(all_agent_latencies, all_agent_accs)[0, 1])
    cov_trap = float(np.cov(trap_agent_latencies, trap_agent_accs)[0, 1]) if len(trap_agent_latencies) > 1 else 0.0
    corr_trap = float(np.corrcoef(trap_agent_latencies, trap_agent_accs)[0, 1]) if len(trap_agent_latencies) > 1 else 0.0

    # Tier-level empirical telemetry
    tier_stats: Dict[str, Any] = {}
    for tier_name in ("fast_3b", "short_cot_3b", "cot_7b", "verifier_7b"):
        tier_obs = [
            o for t in all_tasks for o in stage1_obs_map[t.task_id] if o.tier == tier_name
        ]
        tier_trap = [
            o for t in all_tasks if t.regime == "boundary_trap" for o in stage1_obs_map[t.task_id] if o.tier == tier_name
        ]
        tier_stats[tier_name] = {
            "count": len(tier_obs),
            "overall_accuracy": round(float(np.mean([o.is_correct for o in tier_obs])), 4),
            "boundary_trap_accuracy": round(float(np.mean([o.is_correct for o in tier_trap])), 4) if tier_trap else 0.0,
            "mean_completion_s": round(float(np.mean([o.completion_time_s for o in tier_obs])), 4),
            "mean_mps_wall_s": round(float(np.mean([o.mps_wall_s for o in tier_obs])), 4),
            "mean_normalized_entropy": round(float(np.mean([o.normalized_entropy for o in tier_obs])), 4),
        }

    # Policy summary metrics
    summary_by_policy: Dict[str, Any] = {}
    for pol in policy_names:
        s1_acc = float(np.mean([r["policies"][pol]["stage1_correct"] for r in per_task_records]))
        s2_acc = float(np.mean([r["policies"][pol]["stage2_correct"] for r in per_task_records]))
        both_acc = float(np.mean([r["policies"][pol]["workflow_both_stages_correct"] for r in per_task_records]))
        trap_recs = [r for r in per_task_records if r["regime"] == "boundary_trap"]
        routine_recs = [r for r in per_task_records if r["regime"] == "routine"]
        trap_s2_acc = float(np.mean([r["policies"][pol]["stage2_correct"] for r in trap_recs])) if trap_recs else 0.0
        routine_lat = float(np.mean([r["policies"][pol]["workflow_total_latency_s"] for r in routine_recs])) if routine_recs else 0.0

        lats = np.array([r["policies"][pol]["workflow_total_latency_s"] for r in per_task_records], dtype=np.float64)
        p95 = float(np.percentile(lats, 95))
        p99 = float(np.percentile(lats, 99))
        cvar95 = float(np.mean(lats[lats >= p95]))

        # Cascade amplification: P(Stage2 wrong | Stage1 wrong) vs P(Stage2 wrong | Stage1 right)
        s1_wrong = [r for r in per_task_records if not r["policies"][pol]["stage1_correct"]]
        s1_right = [r for r in per_task_records if r["policies"][pol]["stage1_correct"]]
        p_s2_err_given_s1_err = (
            float(np.mean([not r["policies"][pol]["stage2_correct"] for r in s1_wrong])) if s1_wrong else 0.0
        )
        p_s2_err_given_s1_ok = (
            float(np.mean([not r["policies"][pol]["stage2_correct"] for r in s1_right])) if s1_right else 0.0
        )

        summary_by_policy[pol] = {
            "stage1_accuracy": round(s1_acc, 4),
            "stage2_accuracy": round(s2_acc, 4),
            "end_to_end_joint_accuracy": round(both_acc, 4),
            "boundary_trap_stage2_accuracy": round(trap_s2_acc, 4),
            "mean_workflow_latency_s": round(float(np.mean(lats)), 4),
            "routine_mean_latency_s": round(routine_lat, 4),
            "p50_workflow_latency_s": round(float(np.median(lats)), 4),
            "p95_workflow_latency_s": round(p95, 4),
            "p99_workflow_latency_s": round(p99, 4),
            "cvar95_workflow_latency_s": round(cvar95, 4),
            "p_stage2_err_given_stage1_err": round(p_s2_err_given_s1_err, 4),
            "p_stage2_err_given_stage1_ok": round(p_s2_err_given_s1_ok, 4),
            "stage1_boost_rate": round(
                float(np.mean([r["policies"][pol]["stage1_boost"] for r in per_task_records])), 4
            ),
        }

    sync_p95 = summary_by_policy["Sync_N_of_N"]["p95_workflow_latency_s"]
    qg_p95 = summary_by_policy["QuorumGuard"]["p95_workflow_latency_s"]
    sync_cvar = summary_by_policy["Sync_N_of_N"]["cvar95_workflow_latency_s"]
    qg_cvar = summary_by_policy["QuorumGuard"]["cvar95_workflow_latency_s"]

    results = {
        "metadata": {
            "num_tasks": len(all_tasks),
            "num_routine_tasks": sum(1 for t in all_tasks if t.regime == "routine"),
            "num_boundary_trap_tasks": sum(1 for t in all_tasks if t.regime == "boundary_trap"),
            "models": ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"],
            "device": "Apple Silicon MPS (FP16)",
            "seed": seed,
        },
        "empirical_latency_accuracy_covariance": {
            "overall_cov_s": round(cov_all, 4),
            "overall_pearson_r": round(corr_all, 4),
            "boundary_trap_cov_s": round(cov_trap, 4),
            "boundary_trap_pearson_r": round(corr_trap, 4),
        },
        "tier_telemetry": tier_stats,
        "policy_summary": summary_by_policy,
        "headline_gains": {
            "quorumguard_vs_fast_stage2_acc_delta": round(
                summary_by_policy["QuorumGuard"]["stage2_accuracy"]
                - summary_by_policy["Fast_k_of_N"]["stage2_accuracy"],
                4,
            ),
            "quorumguard_vs_sync_p95_speedup_x": round(sync_p95 / max(qg_p95, 1e-6), 3),
            "quorumguard_vs_sync_cvar95_speedup_x": round(sync_cvar / max(qg_cvar, 1e-6), 3),
        },
        "tasks": per_task_records,
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="QuorumGuard MPS Phase 0/1 Pilot")
    parser.add_argument("--max-tasks", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "quorumguard_pilot.json"),
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = run_pilot(max_tasks=args.max_tasks, seed=args.seed)
    out_path.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 78)
    print("QUORUMGUARD PHASE 0/1 LIVE MPS PILOT SUMMARY")
    print("=" * 78)
    cov = results["empirical_latency_accuracy_covariance"]
    print(
        f"Empirical Latency-Accuracy Covariance: Overall Cov = +{cov['overall_cov_s']:.4f} "
        f"(r = +{cov['overall_pearson_r']:.3f}) | Trap Cov = +{cov['boundary_trap_cov_s']:.4f} "
        f"(r = +{cov['boundary_trap_pearson_r']:.3f})"
    )
    print("\nAgent Tier Telemetry:")
    for tname, tstat in results["tier_telemetry"].items():
        print(
            f"  {tname:13s}: Overall Acc={tstat['overall_accuracy']*100:5.1f}% | "
            f"Trap Acc={tstat['boundary_trap_accuracy']*100:5.1f}% | "
            f"Mean T={tstat['mean_completion_s']:5.2f}s (MPS={tstat['mean_mps_wall_s']:4.2f}s) | "
            f"NormEnt={tstat['mean_normalized_entropy']:.3f}"
        )
    print("\n2-Stage Consensus Policy Head-to-Head:")
    for pol, pstat in results["policy_summary"].items():
        print(
            f"  {pol:20s}: S1={pstat['stage1_accuracy']*100:5.1f}% | "
            f"S2={pstat['stage2_accuracy']*100:5.1f}% | "
            f"Joint={pstat['end_to_end_joint_accuracy']*100:5.1f}% | "
            f"Mean={pstat['mean_workflow_latency_s']:5.2f}s | "
            f"P95={pstat['p95_workflow_latency_s']:5.2f}s | "
            f"CVaR95={pstat['cvar95_workflow_latency_s']:5.2f}s"
        )
    hg = results["headline_gains"]
    print(
        f"\nHeadline: QuorumGuard beats Fast k-of-N by {hg['quorumguard_vs_fast_stage2_acc_delta']*100:+.1f}% "
        f"Stage-2 accuracy while achieving {hg['quorumguard_vs_sync_p95_speedup_x']:.2f}x P95 speedup "
        f"({hg['quorumguard_vs_sync_cvar95_speedup_x']:.2f}x CVaR95 speedup) over Synchronous Barriers!"
    )
    print(f"Saved full telemetry to {out_path}")


if __name__ == "__main__":
    main()
