#!/usr/bin/env python3
"""Static validator for paper/main.tex and paper/references.bib."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> int:
    paper_dir = Path(__file__).resolve().parent
    tex = (paper_dir / "main.tex").read_text()
    bib = (paper_dir / "references.bib").read_text()

    errors: list[str] = []

    # 1. Brace balance (ignoring escaped \{ and \} and comments)
    clean_lines = []
    for line in tex.splitlines():
        line_nc = re.sub(r"(?<!\\)%.*$", "", line)
        clean_lines.append(line_nc)
    clean_tex = "\n".join(clean_lines)

    depth = 0
    for idx, ch in enumerate(clean_tex):
        if ch == "{" and (idx == 0 or clean_tex[idx - 1] != "\\"):
            depth += 1
        elif ch == "}" and (idx == 0 or clean_tex[idx - 1] != "\\"):
            depth -= 1
            if depth < 0:
                errors.append(f"Unmatched closing brace near char offset {idx}")
                break
    if depth != 0:
        errors.append(f"Net brace imbalance: depth={depth}")

    # 2. Environment begin/end stack
    env_stack: list[str] = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", clean_tex):
        kind, env = m.group(1), m.group(2)
        if kind == "begin":
            env_stack.append(env)
        else:
            if not env_stack or env_stack[-1] != env:
                errors.append(f"Mismatched \\end{{{env}}} (stack top: {env_stack[-1] if env_stack else None})")
            else:
                env_stack.pop()
    if env_stack:
        errors.append(f"Unclosed environments: {env_stack}")

    # 3. Citations vs BibTeX keys
    bib_keys = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", bib))
    cited_keys: set[str] = set()
    for m in re.finditer(r"\\cite\{([^}]+)\}", clean_tex):
        for k in m.group(1).split(","):
            cited_keys.add(k.strip())
    missing_cites = sorted(cited_keys - bib_keys)
    if missing_cites:
        errors.append(f"Missing BibTeX keys in references.bib: {missing_cites}")

    # 4. Labels vs refs
    labels = set(re.findall(r"\\label\{([^}]+)\}", clean_tex))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", clean_tex))
    missing_refs = sorted(refs - labels)
    if missing_refs:
        errors.append(f"Undefined \\ref / \\eqref labels: {missing_refs}")

    # 5. Graphics files exist
    figs_referenced = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", clean_tex)
    for f in figs_referenced:
        fpath = paper_dir / "figures" / f
        if not fpath.exists() and not (paper_dir / f).exists():
            errors.append(f"Missing figure file: {f}")

    if errors:
        print("LATEX VALIDATION FAILED:")
        for e in errors:
            print("  -", e)
        return 1

    print(
        f"LATEX VALIDATION PASSED: 0 errors | {len(cited_keys)}/{len(bib_keys)} citations verified | "
        f"{len(labels)} labels | {len(figs_referenced)} figures verified."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
