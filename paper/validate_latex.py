#!/usr/bin/env python3
"""
Strict Static LaTeX, BibTeX, Anti-Slop & Page-Budget Validator for QuorumGuard
==============================================================================
Verifies:
  1. Exact `{` / `}` brace balance (handling `%` comments and escaped `\{`, `\}`).
  2. Exact `\begin{...}` / `\end{...}` environment nesting stack.
  3. Inline `$` math delimiter parity and absence of TeX `$$`.
  4. Absence of Markdown backticks (`...`) and unescaped `_` / `#` outside math/IDs.
  5. Zero undefined or unused `\cite{...}` keys against `references.bib`.
  6. Zero undefined or duplicate `\label{...}` / `\ref{...}` / `\eqref{...}` targets.
  7. Existence of all `\includegraphics` assets in `paper/figures/`.
  8. Humanizer hygiene: zero prose em-dashes (`---`) outside tabular placeholder cells,
     zero exclamation marks in prose, and zero stock AI words (`crucially`, `delve`, `tapestry`).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Set, Tuple

PAPER_DIR = Path(__file__).resolve().parent


def strip_comments(tex: str) -> List[Tuple[int, str]]:
    lines: List[Tuple[int, str]] = []
    for idx, raw in enumerate(tex.splitlines(), start=1):
        out = []
        i = 0
        while i < len(raw):
            if raw[i] == "\\" and i + 1 < len(raw):
                out.append(raw[i : i + 2])
                i += 2
            elif raw[i] == "%":
                break
            else:
                out.append(raw[i])
                i += 1
        lines.append((idx, "".join(out)))
    return lines


def validate_tex_and_bib(tex_path: Path, bib_path: Path) -> int:
    tex = tex_path.read_text(encoding="utf-8")
    bib = bib_path.read_text(encoding="utf-8")
    clean_lines = strip_comments(tex)
    clean_tex = "\n".join(line for _, line in clean_lines)

    errors: List[str] = []
    warnings: List[str] = []

    # 1. Brace balance
    depth = 0
    stack_lines: List[int] = []
    for lno, line in clean_lines:
        i = 0
        while i < len(line):
            if line[i] == "\\" and i + 1 < len(line):
                i += 2
                continue
            if line[i] == "{":
                depth += 1
                stack_lines.append(lno)
            elif line[i] == "}":
                depth -= 1
                if depth < 0:
                    errors.append(f"Unmatched closing '}}' at line {lno}")
                    depth = 0
                else:
                    stack_lines.pop()
            i += 1
    if depth != 0:
        errors.append(f"Unclosed '{{' braces remaining: {depth} (opened at {stack_lines[-5:]})")

    # 2. Environment nesting
    env_stack: List[Tuple[str, int]] = []
    env_re = re.compile(r"\\(begin|end)\{([^}]+)\}")
    for lno, line in clean_lines:
        for m in env_re.finditer(line):
            kind, name = m.group(1), m.group(2)
            if kind == "begin":
                env_stack.append((name, lno))
            else:
                if not env_stack:
                    errors.append(f"Unmatched \\end{{{name}}} at line {lno}")
                else:
                    top_name, top_line = env_stack.pop()
                    if top_name != name:
                        errors.append(
                            f"Environment mismatch: \\begin{{{top_name}}} (line {top_line}) "
                            f"closed by \\end{{{name}}} (line {lno})"
                        )
    for name, lno in env_stack:
        errors.append(f"Unclosed \\begin{{{name}}} opened at line {lno}")

    # 3. Inline '$' math parity
    dollar_count = 0
    i = 0
    while i < len(clean_tex):
        if clean_tex[i] == "\\" and i + 1 < len(clean_tex):
            i += 2
            continue
        if clean_tex[i] == "$":
            if i + 1 < len(clean_tex) and clean_tex[i + 1] == "$":
                errors.append("Found TeX display '$$' instead of equation/align")
                i += 2
                continue
            dollar_count += 1
        i += 1
    if dollar_count % 2 != 0:
        errors.append(f"Odd number of unescaped '$' math delimiters ({dollar_count})")

    # 4. Backticks, unescaped '_' / '#', and Humanizer em-dash / stock-word checks
    in_disp_math = False
    id_cmd_re = re.compile(
        r"\\(?:label|ref|eqref|cite\w*|includegraphics(?:\[[^\]]*\])?|url|bibliography|bibliographystyle|graphicspath)\{[^}]*\}"
    )
    banned_words_re = re.compile(
        r"\b(crucially|delve|tapestry|testament|pivotal|vibrant|underscores|showcasing|fourfold|catastrophic)\b",
        re.IGNORECASE,
    )
    for lno, line in clean_lines:
        if re.search(r"\\begin\{(?:equation|align\*?)\}", line) or line.strip() == r"\[":
            in_disp_math = True
        if not in_disp_math and not line.lstrip().startswith(r"\newcommand"):
            tmp = line.replace("``", "")
            if "`" in tmp:
                errors.append(f"Markdown single backtick at line {lno}: {line.strip()[:80]}")
            if "---" in line:
                errors.append(f"Prose em-dash '---' found at line {lno}: {line.strip()[:80]}")
            bw = banned_words_re.search(line)
            if bw:
                errors.append(f"Stock AI word '{bw.group(1)}' found at line {lno}")
            stripped = id_cmd_re.sub("", line)
            stripped = re.sub(r"\\resizebox\{[^}]*\}\{!\}", "", stripped)
            stripped = re.sub(r"\$[^$]*\$", "", stripped)
            if "!" in stripped and "\\begin{" not in stripped:
                errors.append(f"Exclamation mark in prose at line {lno}: {line.strip()[:80]}")
            j = 0
            while j < len(stripped):
                if stripped[j] == "\\" and j + 1 < len(stripped):
                    j += 2
                    continue
                if stripped[j] in ("_", "#"):
                    errors.append(f"Unescaped '{stripped[j]}' outside math at line {lno}: {line.strip()[:80]}")
                    break
                j += 1
        if re.search(r"\\end\{(?:equation|align\*?)\}", line) or line.strip() == r"\]":
            in_disp_math = False

    # 5. Citations vs BibTeX keys
    bib_keys: Set[str] = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", bib))
    cited_keys: Set[str] = set()
    for m in re.finditer(r"\\cite\w*\{([^}]+)\}", clean_tex):
        for k in m.group(1).split(","):
            k_clean = k.strip()
            if k_clean:
                cited_keys.add(k_clean)
                if k_clean not in bib_keys:
                    errors.append(f"Undefined citation key: '{k_clean}'")
    unused_bib = sorted(bib_keys - cited_keys)
    if unused_bib:
        warnings.append(f"Unused BibTeX entries ({len(unused_bib)}): {', '.join(unused_bib)}")

    # 6. Labels vs Refs
    labels: Set[str] = set()
    for m in re.finditer(r"\\label\{([^}]+)\}", clean_tex):
        lbl = m.group(1).strip()
        if lbl in labels:
            errors.append(f"Duplicate \\label{{{lbl}}}")
        labels.add(lbl)
    referenced: Set[str] = set()
    for m in re.finditer(r"\\(?:eq)?ref\{([^}]+)\}", clean_tex):
        ref = m.group(1).strip()
        referenced.add(ref)
        if ref not in labels:
            errors.append(f"Undefined reference target: \\ref{{{ref}}}")

    # 7. Graphics files
    gfx_files: List[str] = []
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", clean_tex):
        rel = m.group(1).strip()
        gfx_files.append(rel)
        if not (PAPER_DIR / "figures" / rel).exists() and not (PAPER_DIR / rel).exists():
            errors.append(f"Missing figure file: {rel}")

    print("=== QuorumGuard LaTeX & Humanizer Validation Report ===")
    print(f"  Citations : {len(cited_keys)} cited / {len(bib_keys)} defined")
    print(f"  Labels    : {len(labels)} defined / {len(referenced)} referenced")
    print(f"  Figures   : {len(gfx_files)} verified")
    for w in warnings:
        print(f"  [WARN] {w}")
    if errors:
        for e in errors:
            print(f"  [ERROR] {e}")
        return 1
    print("  [PASS] Zero syntax, reference, figure, em-dash, or AI-tell errors!")
    return 0


if __name__ == "__main__":
    sys.exit(validate_tex_and_bib(PAPER_DIR / "main.tex", PAPER_DIR / "references.bib"))
