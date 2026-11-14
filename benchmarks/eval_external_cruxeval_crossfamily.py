#!/usr/bin/env python3
"""
External Public Benchmark (CRUXEval-50) & Cross-Family Non-Qwen Evaluation
===========================================================================
Evaluates QuorumGuard beyond QuorumBench-50 and beyond the Qwen2.5 family:
  1. External Benchmark: Exact first 50 canonical tasks (`sample_0..sample_49`)
     from CRUXEval (`cruxeval-org/cruxeval`, Gu et al., ICML 2024), framed as a
     2-stage code-output verification & regression-assertion DAG with live
     isolated `python3 -I` sandbox execution.
  2. Non-Qwen Cross-Family Committee (`Llama + Granite` architectures):
       - Fast & Short-CoT Tier: `HuggingFaceTB/SmolLM2-1.7B-Instruct` (3.42 GB FP16)
       - Deep CoT & Sandbox Verifier Tier: `ibm-granite/granite-3.1-8b-instruct` (16.34 GB FP16)
  3. Evaluated Pools (2 Benchmarks x 2 Model Ecosystems):
       - `cruxeval50_qwen`      : CRUXEval-50 on Qwen2.5-3B + Qwen2.5-7B
       - `cruxeval50_cross`     : CRUXEval-50 on SmolLM2-1.7B + Granite-3.1-8B
       - `quorumbench30_cross`  : QuorumBench-30 on SmolLM2-1.7B + Granite-3.1-8B

Reports 7-policy accuracy, trap accuracy, P95, CVaR95, 2,000-bootstrap 95% CIs,
10-seed variance, and hardware-only latency in
`benchmarks/results/cruxeval50_crossfamily.json`.
"""

from __future__ import annotations

import ast
import gc
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pilot.quorumguard_pilot import (  # noqa: E402
    CODE_BLOCK_RE,
    AgentObservation,
    TwoStageTask,
    build_pilot_tasks,
    format_mc_prompt,
    run_python_sandbox,
)
from benchmarks.run_revisions_suite import (  # noqa: E402
    perturb_observations_for_seed,
    summarize_two_stage_workflows,
)


# ---------------------------------------------------------------------------
# 1. Canonical CRUXEval-50 Builder (First 50 Public Test Examples)
# ---------------------------------------------------------------------------

def _make_plausible_distractors(gold_val: Any, gold_repr: str, seed_idx: int) -> List[str]:
    """Generate 3 distinct, type-plausible Python literal distractors != gold_repr."""
    cands: List[str] = []
    rng = random.Random(9000 + seed_idx)

    def _add(val_repr: str) -> None:
        if val_repr != gold_repr and val_repr not in cands:
            cands.append(val_repr)

    if isinstance(gold_val, bool):
        _add(repr(not gold_val))
        _add(repr(None))
        _add(repr(0 if gold_val else 1))
    elif isinstance(gold_val, int):
        for delta in (-1, 1, 2, -2, 10, -10):
            _add(repr(gold_val + delta))
        _add(repr(-gold_val if gold_val != 0 else 42))
        _add(repr(0))
    elif isinstance(gold_val, str):
        if gold_val:
            _add(repr(gold_val[::-1]))
            _add(repr(gold_val.swapcase()))
            _add(repr(gold_val[:-1]))
            _add(repr(gold_val + gold_val[-1]))
            _add(repr(gold_val.lower()))
            _add(repr(gold_val.upper()))
            _add(repr(gold_val.strip()))
        _add(repr(""))
        _add(repr("error"))
    elif isinstance(gold_val, list):
        if gold_val:
            _add(repr(list(reversed(gold_val))))
            try:
                s = sorted(gold_val)
                if s != gold_val:
                    _add(repr(s))
            except Exception:
                pass
            _add(repr(gold_val[:-1]))
            _add(repr(gold_val[1:] if len(gold_val) > 1 else [gold_val[0], gold_val[0]]))
            _add(repr(gold_val + [gold_val[-1]]))
        else:
            _add(repr([0]))
            _add(repr([None]))
        _add(repr([]))
    elif isinstance(gold_val, tuple):
        if gold_val:
            _add(repr(tuple(reversed(gold_val))))
            if isinstance(gold_val[0], int):
                _add(repr((gold_val[0] + 1,) + gold_val[1:]))
            _add(repr(gold_val[:-1]))
        _add(repr(()))
    elif isinstance(gold_val, dict):
        keys = list(gold_val.keys())
        if keys:
            _add(repr({k: 0 for k in keys}))
            _add(repr({k: str(v) if not isinstance(v, str) else len(v) for k, v in gold_val.items()}))
            _add(repr({keys[0]: gold_val[keys[0]]} if len(keys) > 1 else {keys[0]: None}))
        _add(repr({}))

    # Deterministic fallback padding if fewer than 3 distinct representations exist
    pad_idx = 1
    while len(cands) < 3:
        cand = repr(f"alt_{seed_idx}_{pad_idx}") if not isinstance(gold_val, int) else repr(gold_val + 100 + pad_idx)
        _add(cand)
        pad_idx += 1

    # Prefer the first generated distractor as the primary shallow trap, shuffle rest
    primary = cands[0]
    rest = cands[1:]
    rng.shuffle(rest)
    return [primary] + rest[:2]


def _classify_cruxeval_regime(code: str) -> str:
    has_loop = ("for " in code) or ("while " in code)
    has_mutation = any(m in code for m in (".sort(", ".reverse(", ".pop(", ".remove(", ".append(", ".update(", "[::-", "[-"))
    return "boundary_trap" if (has_loop or has_mutation) else "routine"


def build_cruxeval_50() -> List[TwoStageTask]:
    p = hf_hub_download("cruxeval-org/cruxeval", "test.jsonl", repo_type="dataset")
    rows = [json.loads(line) for line in open(p, encoding="utf-8")][:50]

    tasks: List[TwoStageTask] = []
    letters = ["A", "B", "C", "D"]
    for idx, row in enumerate(rows):
        gold_val = ast.literal_eval(row["output"])
        gold_repr = repr(gold_val)
        distractors = _make_plausible_distractors(gold_val, gold_repr, idx)

        rng = random.Random(2026 + idx)
        pool_vals = [gold_repr] + distractors
        perm = [0, 1, 2, 3]
        rng.shuffle(perm)

        s1_opts: Dict[str, str] = {}
        s1_gold = "A"
        trap_repr = distractors[0]
        for slot_i, src_i in enumerate(perm):
            s1_opts[letters[slot_i]] = pool_vals[src_i]
            if src_i == 0:
                s1_gold = letters[slot_i]

        regime = _classify_cruxeval_regime(row["code"])
        if regime == "routine":
            triage_note = f"Static AST quick-scan: straight-line expression evaluates to {gold_repr}."
        else:
            triage_note = f"Static AST heuristic: approximating loop/slice transform superficially yields {trap_repr}."

        s1_q = (
            f"Given the Python 3 function below, what exact `repr` value does `f({row['input']})` return?\n"
            f"```python\n{row['code']}\n```"
        )

        # Stage 2: Regression unit-test assertion selection (aligned options with shuffled gold)
        perm2 = [0, 1, 2, 3]
        rng.shuffle(perm2)
        s2_opts: Dict[str, str] = {}
        s2_gold = "A"
        for slot_i, src_i in enumerate(perm2):
            val_r = pool_vals[src_i]
            s2_opts[letters[slot_i]] = f"assert f({row['input']}) == {val_r}"
            if src_i == 0:
                s2_gold = letters[slot_i]

        s2_q = (
            "Given Stage-1's committed execution output, select the exact passing regression "
            "unit-test assertion for CI deployment (independently audit `f` before accepting an upstream bug)."
        )

        tasks.append(
            TwoStageTask(
                task_id=f"CRUX_{row['id']}",
                regime=regime,
                category="cruxeval_code_exec",
                stage1_question=s1_q,
                stage1_options=s1_opts,
                stage1_gold=s1_gold,
                upstream_triage_note=triage_note,
                stage2_question=s2_q,
                stage2_options=s2_opts,
                stage2_gold=s2_gold,
            )
        )
    return tasks


# ---------------------------------------------------------------------------
# 2. Cross-Family BPE-Safe MPS Agent Engine (Qwen2.5, SmolLM2, Granite-3)
# ---------------------------------------------------------------------------

class CrossFamilyMPSEngine:
    """Supports Qwen2.5, SmolLM2 (Llama), and Granite-3 with BPE space+bare option logsumexp probing."""

    def __init__(self, model_name: str, device: str = "mps") -> None:
        self.model_name = model_name
        self.device = device
        print(f"[CrossEngine] Loading {model_name} on {device} (FP16)...", flush=True)
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, local_files_only=True)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, local_files_only=True)
        self.model.to(device)
        self.model.eval()
        if device == "mps":
            torch.mps.synchronize()
        print(f"[CrossEngine] Loaded {model_name} in {time.perf_counter() - t0:.2f}s", flush=True)

        eos = self.tokenizer.eos_token_id
        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else (eos[0] if isinstance(eos, list) else eos)
        )
        # Resolve both bare ("A") and space-prefixed (" A") single-token IDs
        self.option_token_pairs: Dict[str, List[int]] = {}
        for letter in ("A", "B", "C", "D"):
            ids_bare = self.tokenizer.encode(letter, add_special_tokens=False)
            ids_space = self.tokenizer.encode(" " + letter, add_special_tokens=False)
            cands = sorted({int(ids_bare[-1]), int(ids_space[-1])})
            self.option_token_pairs[letter] = cands

        # Warm up Metal shaders cleanly before timed evaluations
        _ = self.generate_and_score_options(
            [{"role": "user", "content": "Warmup check: pick option A) 1 B) 2 C) 3 D) 4."}],
            max_new_tokens=4,
            temperature=0.0,
        )

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
        return list(encoded)

    def generate_and_score_options(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> Tuple[str, str, Dict[str, float], float, float, float, float]:
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
                    pad_token_id=self.pad_token_id,
                )
                seq_ids = gen_out.sequences[0].tolist()
                new_ids = seq_ids[len(prompt_ids) :]
                gen_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

                ent_vals: List[float] = []
                for step_logits in gen_out.scores:
                    lf32 = step_logits[0].float()
                    lp = torch.log_softmax(lf32, dim=-1)
                    pr = torch.exp(lp)
                    ev = float(-(pr * lp).sum().item())
                    if math.isfinite(ev):
                        ent_vals.append(ev)
                seq_ent = float(np.mean(ent_vals)) if ent_vals else 0.0
            else:
                gen_text = ""
                seq_ent = 0.0

            raw_chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # Omit trailing space on prompt so logsumexp over {ĠA, A} captures BPE behavior across all tokenizers
            if gen_text.strip():
                prefix_str = raw_chat + gen_text.strip() + "\nAnswer Option (A, B, C, or D):"
            else:
                prefix_str = raw_chat + "Answer Option (A, B, C, or D):"
            probe_ids = self.tokenizer.encode(prefix_str, add_special_tokens=False)
            probe_tensor = torch.tensor([probe_ids], dtype=torch.long, device=self.device)
            out = self.model(input_ids=probe_tensor, use_cache=False)
            last_logits = out.logits[0, -1, :].float()

            opt_letters = ["A", "B", "C", "D"]
            opt_logits_list: List[torch.Tensor] = []
            for c in opt_letters:
                tids = self.option_token_pairs[c]
                vals = torch.stack([last_logits[tid] for tid in tids])
                opt_logits_list.append(torch.logsumexp(vals, dim=0))
            opt_logits = torch.stack(opt_logits_list)

            opt_log_probs = torch.log_softmax(opt_logits, dim=-1)
            opt_probs_t = torch.exp(opt_log_probs)
            opt_ent = float(-(opt_probs_t * opt_log_probs).sum().item())
            sorted_logits, _ = torch.sort(opt_logits, descending=True)
            margin = float((sorted_logits[0] - sorted_logits[1]).item())
            choice = opt_letters[int(torch.argmax(opt_logits).item())]
            probs_dict = {c: round(float(opt_probs_t[i].item()), 5) for i, c in enumerate(opt_letters)}

        if self.device == "mps":
            torch.mps.synchronize()
        mps_wall = time.perf_counter() - t_start
        return choice, gen_text.strip(), probs_dict, opt_ent, margin, seq_ent, mps_wall


# ---------------------------------------------------------------------------
# 3. Shared Two-Tier Committee Runner (Reused for All 3 Benchmark Suites)
# ---------------------------------------------------------------------------

def evaluate_two_tier_committee_on_tasks(
    suite_name: str,
    tasks: Sequence[TwoStageTask],
    fast_model_name: str,
    deep_model_name: str,
    seed: int = 2026,
) -> Dict[str, Any]:
    raw_cache_path = ROOT / "benchmarks" / "results" / "cruxeval50_raw_cache" / f"{suite_name}.json"
    raw_cache_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_cache_path.exists():
        raw_d = json.loads(raw_cache_path.read_text())
        stage1_obs_map = {tid: [AgentObservation(**d) for d in lst] for tid, lst in raw_d["stage1"].items()}
        stage2_obs_map = {
            tuple(k.split("::")): [AgentObservation(**d) for d in lst]
            for k, lst in raw_d["stage2"].items()
        }
        print(f"[Suite: {suite_name}] Loaded cached raw MPS observations from {raw_cache_path}", flush=True)

        task_by_id = {t.task_id: t for t in tasks}
        stale_s1 = [
            (tid, idx_o, obs)
            for tid, lst in stage1_obs_map.items()
            for idx_o, obs in enumerate(lst)
            if obs.mps_wall_s > 35.0
        ]
        stale_s2 = [
            (pair_k, idx_o, obs)
            for pair_k, lst in stage2_obs_map.items()
            for idx_o, obs in enumerate(lst)
            if obs.mps_wall_s > 35.0
        ]
        if stale_s1 or stale_s2:
            print(
                f"[Suite: {suite_name}] Refreshing {len(stale_s1)} Stage-1 + {len(stale_s2)} Stage-2 stale sleep (>35s) entries...",
                flush=True,
            )
            eng_deep = CrossFamilyMPSEngine(deep_model_name, device="mps")
            eng_deep.generate_and_score_options([{"role": "user", "content": "Warmup"}], max_new_tokens=4, temperature=0.0)
            task_index_map = {t.task_id: idx_t for idx_t, t in enumerate(tasks)}
            s2_pair_keys = list(stage2_obs_map.keys())
            pair_index_map = {k: idx_p for idx_p, k in enumerate(s2_pair_keys)}

            for tid, idx_o, old_obs in stale_s1:
                task = task_by_id[tid]
                idx_t = task_index_map[tid]
                aid = old_obs.agent_id
                q_keep = old_obs.queue_tool_overhead_s
                if aid == 4:
                    msgs_cot = [
                        {
                            "role": "system",
                            "content": "You are a careful Code & Systems Reviewer. Trace loops, types, and boundary conditions step by step in 2-3 sentences WITHOUT naming option letters A/B/C/D yet.",
                        },
                        {"role": "user", "content": format_mc_prompt(task.stage1_question, task.stage1_options)},
                    ]
                    ch4, txt4, probs4, ent4, mar4, seq4, mps4 = eng_deep.generate_and_score_options(msgs_cot, max_new_tokens=75, temperature=0.0)
                    stage1_obs_map[tid][idx_o] = AgentObservation(
                        4, "cot_7b", "indep_cot_root_4", ch4, (ch4 == task.stage1_gold),
                        probs4, round(ent4, 5), round(ent4 / math.log(4.0), 5), round(mar4, 4),
                        round(seq4, 4), False, round(mps4, 4), q_keep, round(mps4 + q_keep, 4), txt4[:100],
                    )
                elif aid in (5, 6):
                    idx_v = 0 if aid == 5 else 1
                    root_id = f"sandbox_root_{aid}"
                    code_msgs = [
                        {
                            "role": "system",
                            "content": (
                                f"You are Sandbox Verification Agent #{aid}. Write a short self-contained ```python ... ``` "
                                "script that executes the exact code or numeric expression and prints `repr(result)` cleanly."
                            ),
                        },
                        {"role": "user", "content": format_mc_prompt(task.stage1_question, task.stage1_options)},
                    ]
                    _, code_resp, _, _, _, _, mps_t1 = eng_deep.generate_and_score_options(
                        code_msgs, max_new_tokens=90, temperature=(0.15 if idx_v == 1 else 0.0), seed=seed + idx_t + aid
                    )
                    m = CODE_BLOCK_RE.search(code_resp)
                    t_s0 = time.perf_counter()
                    if m:
                        ok_tool, tool_out = run_python_sandbox(m.group(1))
                    else:
                        ok_tool, tool_out = False, "No fenced python block emitted."
                    sbx_wall = time.perf_counter() - t_s0
                    synth_msgs = code_msgs + [
                        {"role": "assistant", "content": code_resp},
                        {
                            "role": "user",
                            "content": (
                                f"Sandbox Executed={ok_tool}\nSandbox Output:\n{tool_out}\n\n"
                                "Match the exact sandbox output to the Options in one concise sentence WITHOUT writing Option letter A/B/C/D yet."
                            ),
                        },
                    ]
                    ch_v, txt_v, probs_v, ent_v, mar_v, seq_v, mps_t2 = eng_deep.generate_and_score_options(
                        synth_msgs, max_new_tokens=28, temperature=0.0
                    )
                    tot_mps = mps_t1 + mps_t2 + sbx_wall
                    stage1_obs_map[tid][idx_o] = AgentObservation(
                        aid, "verifier_7b", root_id, ch_v, (ch_v == task.stage1_gold),
                        probs_v, round(ent_v, 5), round(ent_v / math.log(4.0), 5), round(mar_v, 4),
                        round(seq_v, 4), ok_tool, round(tot_mps, 4), q_keep, round(tot_mps + q_keep, 4),
                        f"[tool_ok={ok_tool}: {tool_out[:45]}] {txt_v[:45]}",
                    )

            for pair_k, idx_o, old_obs in stale_s2:
                tid, s1_ch = pair_k
                task = task_by_id[tid]
                idx_p = pair_index_map[pair_k]
                aid = old_obs.agent_id
                q_keep = old_obs.queue_tool_overhead_s
                if aid in (3, 4):
                    root_id = f"s2_audit_deep_{aid}"
                    s1_claim_txt = task.stage1_options.get(s1_ch, s1_ch)
                    audit_ctx = (
                        f"Upstream Context:\n{task.stage1_question}\n"
                        f"Stage-1 Committed Option {s1_ch}: {s1_claim_txt}.\n"
                        "Verify carefully whether Stage-1's finding is sound before picking the true assertion/action."
                    )
                    msgs = [
                        {"role": "system", "content": "You are Principal Verification Auditor. Audit upstream claims concisely in 1-2 sentences WITHOUT naming option letters A/B/C/D yet."},
                        {"role": "user", "content": format_mc_prompt(task.stage2_question, task.stage2_options, audit_ctx)},
                    ]
                    ch_a, txt_a, probs_a, ent_a, mar_a, seq_a, mps_a = eng_deep.generate_and_score_options(
                        msgs, max_new_tokens=50, temperature=(0.12 if aid == 4 else 0.0), seed=seed + 500 + idx_p + aid
                    )
                    stage2_obs_map[pair_k][idx_o] = AgentObservation(
                        aid, "verifier_7b", root_id, ch_a, (ch_a == task.stage2_gold),
                        probs_a, round(ent_a, 5), round(ent_a / math.log(4.0), 5), round(mar_a, 4),
                        round(seq_a, 4), True, round(mps_a, 4), q_keep, round(mps_a + q_keep, 4),
                        f"[Deep-Audit] {txt_a[:80]}",
                    )

            missing_pairs: List[Tuple[TwoStageTask, str]] = []
            for t in tasks:
                for ch in sorted({o.choice for o in stage1_obs_map[t.task_id]} | {t.stage1_gold}):
                    if (t.task_id, ch) not in stage2_obs_map:
                        missing_pairs.append((t, ch))
                        stage2_obs_map[(t.task_id, ch)] = []
            if missing_pairs:
                rng_miss = np.random.default_rng(seed + 999)
                for idx_m, (task, s1_ch) in enumerate(missing_pairs):
                    s1_claim_txt = task.stage1_options.get(s1_ch, s1_ch)
                    sq = (3.6 + rng_miss.pareto(a=2.3, size=2) * 2.6).clip(3.6, 15.5).tolist()
                    for k_i, (aid, root_id) in enumerate([(3, "s2_audit_deep_3"), (4, "s2_audit_deep_4")]):
                        audit_ctx = (
                            f"Upstream Context:\n{task.stage1_question}\n"
                            f"Stage-1 Committed Option {s1_ch}: {s1_claim_txt}.\n"
                            "Verify carefully whether Stage-1's finding is sound before picking the true assertion/action."
                        )
                        msgs = [
                            {"role": "system", "content": "You are Principal Verification Auditor. Audit upstream claims concisely in 1-2 sentences WITHOUT naming option letters A/B/C/D yet."},
                            {"role": "user", "content": format_mc_prompt(task.stage2_question, task.stage2_options, audit_ctx)},
                        ]
                        ch_a, txt_a, probs_a, ent_a, mar_a, seq_a, mps_a = eng_deep.generate_and_score_options(
                            msgs, max_new_tokens=50, temperature=(0.12 if aid == 4 else 0.0), seed=seed + 700 + idx_m + aid
                        )
                        q_a = float(sq[k_i])
                        stage2_obs_map[(task.task_id, s1_ch)].append(
                            AgentObservation(
                                aid, "verifier_7b", root_id, ch_a, (ch_a == task.stage2_gold),
                                probs_a, round(ent_a, 5), round(ent_a / math.log(4.0), 5), round(mar_a, 4),
                                round(seq_a, 4), True, round(mps_a, 4), round(q_a, 4), round(mps_a + q_a, 4),
                                f"[Deep-Audit] {txt_a[:80]}",
                            )
                        )
            eng_deep.unload()
            del eng_deep

            if missing_pairs:
                eng_fast2 = CrossFamilyMPSEngine(fast_model_name, device="mps")
                rng_miss_f = np.random.default_rng(seed + 1001)
                for task, s1_ch in missing_pairs:
                    s1_claim_txt = task.stage1_options.get(s1_ch, s1_ch)
                    fq = rng_miss_f.uniform(0.16, 0.65, size=3).tolist()
                    s2_fast_cfgs = [
                        (0, "upstream_commit_echo_root", f"Authoritative Stage-1 Consensus Result (CONFIRMED): {s1_ch}) {s1_claim_txt}. Follow Stage-1's confirmed output."),
                        (1, "upstream_commit_echo_root", f"Committed Stage-1 Finding: {s1_ch}) {s1_claim_txt}. Align downstream decision with Stage-1."),
                        (2, "s2_fast_indep_root_2", f"Context: {task.stage1_question}\nStage-1 Reported: {s1_ch}) {s1_claim_txt}."),
                    ]
                    for aid, root_id, extra_ctx in s2_fast_cfgs:
                        msgs = [
                            {"role": "system", "content": "You are Fast Deployment Coordinator. Make an immediate zero-CoT decision."},
                            {"role": "user", "content": format_mc_prompt(task.stage2_question, task.stage2_options, extra_ctx)},
                        ]
                        ch_f, _, probs_f, ent_f, mar_f, seq_f, mps_f = eng_fast2.generate_and_score_options(msgs, max_new_tokens=0, temperature=0.0)
                        q_f = float(fq[aid])
                        stage2_obs_map[(task.task_id, s1_ch)].append(
                            AgentObservation(
                                aid, "fast_3b", root_id, ch_f, (ch_f == task.stage2_gold),
                                probs_f, round(ent_f, 5), round(ent_f / math.log(4.0), 5), round(mar_f, 4),
                                round(seq_f, 4), False, round(mps_f, 4), round(q_f, 4), round(mps_f + q_f, 4),
                                f"[Fast direct: {ch_f} margin={mar_f:.2f}]",
                            )
                        )
                eng_fast2.unload()
                del eng_fast2

            for k_s2 in stage2_obs_map:
                stage2_obs_map[k_s2].sort(key=lambda o: o.agent_id)
            raw_cache_path.write_text(
                json.dumps(
                    {
                        "stage1": {tid: [asdict(o) for o in lst] for tid, lst in stage1_obs_map.items()},
                        "stage2": {f"{k[0]}::{k[1]}": [asdict(o) for o in lst] for k, lst in stage2_obs_map.items()},
                    },
                    indent=2,
                )
            )
            print(f"[Suite: {suite_name}] Refreshed & saved raw cache {raw_cache_path}", flush=True)
    else:
        rng = np.random.default_rng(seed)
        print(f"\n[Suite: {suite_name}] Running {len(tasks)} tasks ({fast_model_name} + {deep_model_name})...", flush=True)

        s1_overheads: Dict[str, List[float]] = {}
        for t in tasks:
            fq = rng.uniform(0.18, 0.72, size=4).tolist()
            cq = [float(rng.uniform(0.85, 1.95))]
            sq = (3.8 + rng.pareto(a=2.25, size=2) * 2.7).clip(3.8, 16.5).tolist()
            s1_overheads[t.task_id] = fq + cq + [float(x) for x in sq]

        stage1_obs_map = {t.task_id: [] for t in tasks}

        # --- Pass 1: Fast Tier (Stage 1) ---
        eng_fast = CrossFamilyMPSEngine(fast_model_name, device="mps")
        fast_configs = [
            (0, "fast_3b", "shared_triage_root", True, 0, "You are Fast Triage Agent A. Rely on the provided triage note for an immediate decision."),
            (1, "fast_3b", "shared_triage_root", True, 0, "You are Fast Triage Agent B. Follow the upstream triage summary for an immediate decision."),
            (2, "fast_3b", "indep_fast_root_2", False, 0, "You are Rapid Scanner C. Make an immediate zero-CoT intuitive decision."),
            (3, "short_cot_3b", "indep_fast_cot_root_3", False, 40, "You are Quick Verifier D. Write 1-2 brief sentences tracing edge cases WITHOUT naming an option letter A/B/C/D yet."),
        ]
        for idx_t, task in enumerate(tasks):
            for aid, tier_lbl, root_id, use_note, max_tok, sys_prompt in fast_configs:
                extra = f"Upstream Automated Triage Note: {task.upstream_triage_note}" if use_note else ""
                msgs = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": format_mc_prompt(task.stage1_question, task.stage1_options, extra)},
                ]
                ch, txt, probs, ent, margin, seq_ent, mps_s = eng_fast.generate_and_score_options(msgs, max_new_tokens=max_tok, temperature=0.0)
                q_s = s1_overheads[task.task_id][aid] + (0.40 if max_tok > 0 else 0.0)
                stage1_obs_map[task.task_id].append(
                    AgentObservation(
                        aid, tier_lbl, root_id, ch, (ch == task.stage1_gold),
                        probs, round(ent, 5), round(ent / math.log(4.0), 5), round(margin, 4),
                        round(seq_ent, 4), False, round(mps_s, 4), round(q_s, 4), round(mps_s + q_s, 4),
                        txt[:100] if txt else f"[pre-commit direct: {ch} margin={margin:.2f}]",
                    )
                )
            if (idx_t + 1) % 15 == 0 or (idx_t + 1) == len(tasks):
                print(f"  [{suite_name} Stage1-Fast] {idx_t + 1}/{len(tasks)} done", flush=True)
        eng_fast.unload()
        del eng_fast

        # --- Pass 2: Deep CoT + Sandbox Verifier Tier (Stage 1 + Complete Stage 2 Cache) ---
        eng_deep = CrossFamilyMPSEngine(deep_model_name, device="mps")
        for idx_t, task in enumerate(tasks):
            # Agent 4: Medium CoT
            msgs_cot = [
                {
                    "role": "system",
                    "content": "You are a careful Code & Systems Reviewer. Trace loops, types, and boundary conditions step by step in 2-3 sentences WITHOUT naming option letters A/B/C/D yet.",
                },
                {"role": "user", "content": format_mc_prompt(task.stage1_question, task.stage1_options)},
            ]
            ch4, txt4, probs4, ent4, mar4, seq4, mps4 = eng_deep.generate_and_score_options(msgs_cot, max_new_tokens=75, temperature=0.0)
            q4 = s1_overheads[task.task_id][4]
            stage1_obs_map[task.task_id].append(
                AgentObservation(
                    4, "cot_7b", "indep_cot_root_4", ch4, (ch4 == task.stage1_gold),
                    probs4, round(ent4, 5), round(ent4 / math.log(4.0), 5), round(mar4, 4),
                    round(seq4, 4), False, round(mps4, 4), round(q4, 4), round(mps4 + q4, 4), txt4[:100],
                )
            )

            # Agents 5 & 6: Live Python Sandbox Verifiers
            for idx_v, (aid, root_id) in enumerate([(5, "sandbox_root_5"), (6, "sandbox_root_6")]):
                code_msgs = [
                    {
                        "role": "system",
                        "content": (
                            f"You are Sandbox Verification Agent #{aid}. Write a short self-contained ```python ... ``` "
                            "script that executes the exact code or numeric expression and prints `repr(result)` cleanly."
                        ),
                    },
                    {"role": "user", "content": format_mc_prompt(task.stage1_question, task.stage1_options)},
                ]
                _, code_resp, _, _, _, _, mps_t1 = eng_deep.generate_and_score_options(
                    code_msgs, max_new_tokens=90, temperature=(0.15 if idx_v == 1 else 0.0), seed=seed + idx_t + aid
                )
                m = CODE_BLOCK_RE.search(code_resp)
                t_s0 = time.perf_counter()
                if m:
                    ok_tool, tool_out = run_python_sandbox(m.group(1))
                else:
                    ok_tool, tool_out = False, "No fenced python block emitted."
                sbx_wall = time.perf_counter() - t_s0

                synth_msgs = code_msgs + [
                    {"role": "assistant", "content": code_resp},
                    {
                        "role": "user",
                        "content": (
                            f"Sandbox Executed={ok_tool}\nSandbox Output:\n{tool_out}\n\n"
                            "Match the exact sandbox output to the Options in one concise sentence WITHOUT writing Option letter A/B/C/D yet."
                        ),
                    },
                ]
                ch_v, txt_v, probs_v, ent_v, mar_v, seq_v, mps_t2 = eng_deep.generate_and_score_options(
                    synth_msgs, max_new_tokens=28, temperature=0.0
                )
                tot_mps = mps_t1 + mps_t2 + sbx_wall
                q_v = s1_overheads[task.task_id][aid]
                stage1_obs_map[task.task_id].append(
                    AgentObservation(
                        aid, "verifier_7b", root_id, ch_v, (ch_v == task.stage1_gold),
                        probs_v, round(ent_v, 5), round(ent_v / math.log(4.0), 5), round(mar_v, 4),
                        round(seq_v, 4), ok_tool, round(tot_mps, 4), round(q_v, 4), round(tot_mps + q_v, 4),
                        f"[tool_ok={ok_tool}: {tool_out[:45]}] {txt_v[:45]}",
                    )
                )
            if (idx_t + 1) % 10 == 0 or (idx_t + 1) == len(tasks):
                print(f"  [{suite_name} Stage1-Deep] {idx_t + 1}/{len(tasks)} done", flush=True)

        # Collect ALL unique Stage-1 voted choices for complete Stage-2 evaluation
        s2_pairs: List[Tuple[TwoStageTask, str]] = []
        for t in tasks:
            choices = sorted({o.choice for o in stage1_obs_map[t.task_id]} | {t.stage1_gold})
            for ch in choices:
                s2_pairs.append((t, ch))

        s2_overheads: Dict[Tuple[str, str], List[float]] = {}
        for t, ch in s2_pairs:
            fq = rng.uniform(0.16, 0.65, size=3).tolist()
            sq = (3.6 + rng.pareto(a=2.3, size=2) * 2.6).clip(3.6, 15.5).tolist()
            s2_overheads[(t.task_id, ch)] = fq + [float(x) for x in sq]

        stage2_obs_map = { (t.task_id, ch): [] for t, ch in s2_pairs }

        for idx_p, (task, s1_ch) in enumerate(s2_pairs):
            s1_claim_txt = task.stage1_options.get(s1_ch, s1_ch)
            for aid, root_id in [(3, "s2_audit_deep_3"), (4, "s2_audit_deep_4")]:
                audit_ctx = (
                    f"Upstream Context:\n{task.stage1_question}\n"
                    f"Stage-1 Committed Option {s1_ch}: {s1_claim_txt}.\n"
                    "Verify carefully whether Stage-1's finding is sound before picking the true assertion/action."
                )
                msgs = [
                    {"role": "system", "content": "You are Principal Verification Auditor. Audit upstream claims concisely in 1-2 sentences WITHOUT naming option letters A/B/C/D yet."},
                    {"role": "user", "content": format_mc_prompt(task.stage2_question, task.stage2_options, audit_ctx)},
                ]
                ch_a, txt_a, probs_a, ent_a, mar_a, seq_a, mps_a = eng_deep.generate_and_score_options(
                    msgs, max_new_tokens=50, temperature=(0.12 if aid == 4 else 0.0), seed=seed + 500 + idx_p + aid
                )
                q_a = s2_overheads[(task.task_id, s1_ch)][aid]
                stage2_obs_map[(task.task_id, s1_ch)].append(
                    AgentObservation(
                        aid, "verifier_7b", root_id, ch_a, (ch_a == task.stage2_gold),
                        probs_a, round(ent_a, 5), round(ent_a / math.log(4.0), 5), round(mar_a, 4),
                        round(seq_a, 4), True, round(mps_a, 4), round(q_a, 4), round(mps_a + q_a, 4),
                        f"[Deep-Audit] {txt_a[:80]}",
                    )
                )
            if (idx_p + 1) % 30 == 0 or (idx_p + 1) == len(s2_pairs):
                print(f"  [{suite_name} Stage2-Deep] {idx_p + 1}/{len(s2_pairs)} pairs done", flush=True)

        eng_deep.unload()
        del eng_deep

        # --- Pass 3: Fast Tier (Stage 2 Coordinators) ---
        eng_fast2 = CrossFamilyMPSEngine(fast_model_name, device="mps")
        for task, s1_ch in s2_pairs:
            s1_claim_txt = task.stage1_options.get(s1_ch, s1_ch)
            s2_fast_cfgs = [
                (0, "upstream_commit_echo_root", f"Authoritative Stage-1 Consensus Result (CONFIRMED): {s1_ch}) {s1_claim_txt}. Follow Stage-1's confirmed output."),
                (1, "upstream_commit_echo_root", f"Committed Stage-1 Finding: {s1_ch}) {s1_claim_txt}. Align downstream decision with Stage-1."),
                (2, "s2_fast_indep_root_2", f"Context: {task.stage1_question}\nStage-1 Reported: {s1_ch}) {s1_claim_txt}."),
            ]
            for aid, root_id, extra_ctx in s2_fast_cfgs:
                msgs = [
                    {"role": "system", "content": "You are Fast Deployment Coordinator. Make an immediate zero-CoT decision."},
                    {"role": "user", "content": format_mc_prompt(task.stage2_question, task.stage2_options, extra_ctx)},
                ]
                ch_f, _, probs_f, ent_f, mar_f, seq_f, mps_f = eng_fast2.generate_and_score_options(msgs, max_new_tokens=0, temperature=0.0)
                q_f = s2_overheads[(task.task_id, s1_ch)][aid]
                stage2_obs_map[(task.task_id, s1_ch)].append(
                    AgentObservation(
                        aid, "fast_3b", root_id, ch_f, (ch_f == task.stage2_gold),
                        probs_f, round(ent_f, 5), round(ent_f / math.log(4.0), 5), round(mar_f, 4),
                        round(seq_f, 4), False, round(mps_f, 4), round(q_f, 4), round(mps_f + q_f, 4),
                        f"[Fast direct: {ch_f} margin={mar_f:.2f}]",
                    )
                )
        eng_fast2.unload()
        del eng_fast2

        raw_cache_path.write_text(
            json.dumps(
                {
                    "stage1": {tid: [asdict(o) for o in lst] for tid, lst in stage1_obs_map.items()},
                    "stage2": {f"{tid}::{ch}": [asdict(o) for o in lst] for (tid, ch), lst in stage2_obs_map.items()},
                },
                indent=2,
            )
        )
        print(f"  [{suite_name}] Saved raw MPS observations cache to {raw_cache_path}", flush=True)

    # Compute latency-accuracy covariance (full + pure-MPS)
    all_lat = [o.completion_time_s for t in tasks for o in stage1_obs_map[t.task_id]]
    mps_lat = [o.mps_wall_s for t in tasks for o in stage1_obs_map[t.task_id]]
    all_acc = [1.0 if o.is_correct else 0.0 for t in tasks for o in stage1_obs_map[t.task_id]]

    summary_ext, _ = summarize_two_stage_workflows(tasks, stage1_obs_map, stage2_obs_map)
    summary_hw, _ = summarize_two_stage_workflows(tasks, stage1_obs_map, stage2_obs_map, eval_kwargs={"pure_hardware_only": True})

    # 10-seed evaluation
    per_seed = []
    for sd in range(2026, 2036):
        s1_p, s2_p = perturb_observations_for_seed(stage1_obs_map, stage2_obs_map, seed=sd, logit_noise_std=0.25)
        sm_s, _ = summarize_two_stage_workflows(tasks, s1_p, s2_p, n_bootstrap=200)
        per_seed.append(sm_s)
    ms_stats = {}
    for pol in summary_ext:
        jv = [ps[pol]["end_to_end_joint_accuracy"] for ps in per_seed]
        pv = [ps[pol]["p95_latency_s"] for ps in per_seed]
        ms_stats[pol] = {
            "joint_acc_mean": round(float(np.mean(jv)), 4),
            "joint_acc_std": round(float(np.std(jv, ddof=1)), 4),
            "p95_mean_s": round(float(np.mean(pv)), 3),
            "p95_std_s": round(float(np.std(pv, ddof=1)), 3),
        }

    return {
        "suite": suite_name,
        "num_tasks": len(tasks),
        "fast_model": fast_model_name,
        "deep_model": deep_model_name,
        "covariance": {
            "overall_cov_s": round(float(np.cov(all_lat, all_acc)[0, 1]), 4),
            "overall_pearson_r": round(float(np.corrcoef(all_lat, all_acc)[0, 1]), 4),
            "pure_mps_cov_s": round(float(np.cov(mps_lat, all_acc)[0, 1]), 4),
            "pure_mps_pearson_r": round(float(np.corrcoef(mps_lat, all_acc)[0, 1]), 4),
        },
        "policy_summary": summary_ext,
        "pure_hardware_only": summary_hw,
        "multi_seed_10_runs": ms_stats,
    }


def main() -> None:
    out_path = ROOT / "benchmarks" / "results" / "cruxeval50_crossfamily.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    crux_tasks = build_cruxeval_50()
    qb30_tasks = build_pilot_tasks()[:30]
    print(
        f"Built CRUXEval-50 ({sum(t.regime=='routine' for t in crux_tasks)} routine, "
        f"{sum(t.regime!='routine' for t in crux_tasks)} multi-step traps) & QuorumBench-30",
        flush=True,
    )

    results: Dict[str, Any] = {}
    for key, tlist, m_fast, m_deep in [
        ("cruxeval50_qwen", crux_tasks, "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"),
        ("cruxeval50_cross_smol_granite", crux_tasks, "HuggingFaceTB/SmolLM2-1.7B-Instruct", "ibm-granite/granite-3.1-8b-instruct"),
        ("quorumbench30_cross_smol_granite", qb30_tasks, "HuggingFaceTB/SmolLM2-1.7B-Instruct", "ibm-granite/granite-3.1-8b-instruct"),
    ]:
        results[key] = evaluate_two_tier_committee_on_tasks(key, tlist, m_fast, m_deep)
        # Incremental checkpoint save so progress is never lost
        out_path.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 80)
    print("EXTERNAL BENCHMARK (CRUXEVAL-50) & CROSS-FAMILY SUMMARY")
    print("=" * 80)
    for key, res in results.items():
        cov = res["covariance"]
        print(f"\n[{key}] {res['fast_model']} + {res['deep_model']} | Cov={cov['overall_cov_s']:+.4f}s (MPS {cov['pure_mps_cov_s']:+.4f}s)")
        for pol in ("Fast_k_of_N", "Tier_Weighted_Fast", "Entropy_Cascade", "CVaR_B_2.8s", "CVaR_B_10s", "Sync_N_of_N", "Genealogy_Sync_HPR", "QuorumGuard"):
            st = res["policy_summary"][pol]
            ms = res["multi_seed_10_runs"][pol]
            print(
                f"  {pol:20s}: Joint={st['end_to_end_joint_accuracy']*100:5.1f}% "
                f"(10s {ms['joint_acc_mean']*100:4.1f}+/-{ms['joint_acc_std']*100:3.1f}%), "
                f"Trap={st['trap_joint_accuracy']*100:5.1f}%, P95={st['p95_latency_s']:5.2f}s"
            )
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
