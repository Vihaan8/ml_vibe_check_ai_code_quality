"""
feature_extraction.py
Phase 1: Feature extraction for Vibe Check

Features match exactly what the proposal describes:

  1. Classical software metrics   (LOC, cyclomatic complexity, nesting depth)
  2. AST structural features      (control flow, error handling, import node counts)
  3. Prompt-code alignment        (library coverage, missing libs, length ratio)
  4. LLM smell features           (hardcoded returns, placeholders, suspiciously short)
"""

import ast
import re
import textwrap
from typing import Optional
import pandas as pd
from radon.complexity import cc_visit
from radon.raw import analyze


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_parse(code: str) -> Optional[ast.Module]:
    """Parse code into an AST. Returns None on syntax error or non-string input."""
    if not isinstance(code, str):
        return None
    try:
        return ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return None


def _ensure_str(value) -> str:
    """Convert NaN / None to empty string."""
    return value if isinstance(value, str) else ""


# ---------------------------------------------------------------------------
# 1. Classical software metrics
# ---------------------------------------------------------------------------

def _classical_features(code: str, tree: Optional[ast.Module]) -> dict:
    """
    LOC        : total physical lines (via radon.raw.analyze)
    CC         : cyclomatic complexity summed across all functions (via radon cc_visit)
    Nesting    : maximum control-flow nesting depth (via ast)
    """
    # --- LOC via radon ---
    loc = 0
    if code.strip():
        try:
            loc = analyze(code).sloc   # source lines of code (non-blank, non-comment)
        except SyntaxError:
            loc = len([l for l in code.splitlines() if l.strip()])

    # --- Cyclomatic complexity via radon ---
    cc = 0
    if code.strip():
        try:
            blocks = cc_visit(code)
            cc = sum(b.complexity for b in blocks) if blocks else 1
        except Exception:
            cc = 0

    # --- Max nesting depth via ast ---
    nesting = 0
    if tree is not None:
        nesting_types = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.ExceptHandler)

        def _depth(node, current=0):
            if isinstance(node, nesting_types):
                current += 1
            children = [_depth(child, current) for child in ast.iter_child_nodes(node)]
            return max([current] + children)

        nesting = _depth(tree)

    return {
        "classical_loc":                    loc,
        "classical_cyclomatic_complexity":  cc,
        "classical_max_nesting_depth":      nesting,
    }


# ---------------------------------------------------------------------------
# 2. AST structural features
# ---------------------------------------------------------------------------

def _ast_features(tree: Optional[ast.Module]) -> dict:
    """
    Count key AST node types.
    Focus is on nodes the proposal explicitly calls out:
    control flow, error handling, imports, returns.
    """
    counts = {
        "ast_if_count":       0,
        "ast_for_count":      0,
        "ast_while_count":    0,
        "ast_try_count":      0,
        "ast_except_count":   0,
        "ast_return_count":   0,
        "ast_import_count":   0,
        "ast_has_error_handling": 0,   # 1 if any try/except exists
    }
    if tree is None:
        return counts

    for node in ast.walk(tree):
        if isinstance(node, ast.If):           counts["ast_if_count"]     += 1
        elif isinstance(node, ast.For):        counts["ast_for_count"]    += 1
        elif isinstance(node, ast.While):      counts["ast_while_count"]  += 1
        elif isinstance(node, ast.Try):        counts["ast_try_count"]    += 1
        elif isinstance(node, ast.ExceptHandler): counts["ast_except_count"] += 1
        elif isinstance(node, ast.Return):     counts["ast_return_count"] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            counts["ast_import_count"] += 1

    counts["ast_has_error_handling"] = int(counts["ast_try_count"] > 0)
    return counts


# ---------------------------------------------------------------------------
# 3. Prompt-code alignment features
# ---------------------------------------------------------------------------

# The choice of the libs are using the ones are commonly used in Python coding tasks and 
# are likely to be mentioned in prompts. 
_LIB_ALIASES = {
    "numpy": ["numpy", "np"],
    "pandas": ["pandas", "pd"],
    "matplotlib": ["matplotlib", "plt", "mpl"],
    "sklearn": ["sklearn", "scikit"],
    "scipy": ["scipy"],
    "requests": ["requests"],
    "json": ["json"],
    "os": ["os"],
    "sys": ["sys"],
    "re": ["re"],
    "datetime": ["datetime"],
    "pathlib": ["pathlib"],
    "collections": ["collections"],
    "itertools": ["itertools"],
    "math": ["math"],
    "bisect": ["bisect"]
}



def _imported_libs(tree: Optional[ast.Module]) -> set:
    if tree is None:
        return set()
    libs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                libs.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            libs.add(node.module.split(".")[0])
    return libs


def _prompt_libs(prompt: str) -> set:
    prompt_lower = prompt.lower()
    return {
        lib for lib, aliases in _LIB_ALIASES.items()
        if any(a in prompt_lower for a in aliases)
    }


def _alignment_features(code: str, prompt: str, tree: Optional[ast.Module]) -> dict:
    """
    lib_coverage  : fraction of prompt-mentioned libraries actually imported
    missing_libs  : count of prompt-mentioned libraries NOT imported
    length_ratio  : len(code) / len(prompt) — low ratio suggests incomplete response
    """
    imported  = _imported_libs(tree)
    mentioned = _prompt_libs(prompt)

    lib_coverage = (
        len(imported & mentioned) / len(mentioned) if mentioned else 1.0
    )

    return {
        "align_lib_coverage":   round(lib_coverage, 4),
        "align_missing_libs":   len(mentioned - imported),
        "align_length_ratio":   round(len(code) / max(len(prompt), 1), 4),
    }


# ---------------------------------------------------------------------------
# 4. LLM smell features
# ---------------------------------------------------------------------------

_PLACEHOLDER_PATTERNS = [
    r"\bpass\b",
    r"\.\.\.",
    r"#\s*TODO",
    r"#\s*FIXME",
    r"raise\s+NotImplementedError",
    r"your\s+code\s+here",
]


def _smell_features(code: str, tree: Optional[ast.Module]) -> dict:
    """
    hardcoded_return  : functions whose entire body is return <literal>
    placeholder_hits  : count of pass / ... / TODO / NotImplementedError
    is_very_short     : 1 if ≤ 5 non-blank lines (proposal: below 10th percentile)
    """
    # Hardcoded return: function body is just `return <constant>`
    hardcoded_returns = 0
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                real = [n for n in node.body if not isinstance(n, ast.Expr)]
                if (len(real) == 1
                        and isinstance(real[0], ast.Return)
                        and isinstance(real[0].value, ast.Constant)):
                    hardcoded_returns += 1

    # Placeholder patterns
    placeholder_hits = sum(
        len(re.findall(pat, code, re.IGNORECASE | re.MULTILINE))
        for pat in _PLACEHOLDER_PATTERNS
    )

    # Suspiciously short
    non_blank = [l for l in code.splitlines() if l.strip()]
    is_very_short = int(len(non_blank) <= 5)

    return {
        "smell_hardcoded_return_funcs": hardcoded_returns,
        "smell_placeholder_hits":       placeholder_hits,
        "smell_is_very_short":          is_very_short,
    }


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def extract_features(code: str, prompt: str = "") -> dict:
    """
    Extract all Phase 1 features for a single (code, prompt) pair.

    Parameters
    ----------
    code   : AI-generated source code string
    prompt : instruct prompt / task description (can be empty)

    Returns
    -------
    dict of 17 features + 1 meta field (parse error flag)
    """
    code   = _ensure_str(code)
    prompt = _ensure_str(prompt)
    tree   = _safe_parse(code)

    feats = {"meta_parse_error": int(tree is None)}
    feats.update(_classical_features(code, tree))
    feats.update(_ast_features(tree))
    feats.update(_alignment_features(code, prompt, tree))
    feats.update(_smell_features(code, tree))
    return feats


def extract_features_batch(
    df: pd.DataFrame,
    code_col: str = "generated_code",
    prompt_col: str = "prompt",
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Extract features for every row in a DataFrame.

    Parameters
    ----------
    df            : input DataFrame
    code_col      : column containing source code strings
    prompt_col    : column containing prompt strings
    show_progress : print progress every 5000 rows

    Returns
    -------
    DataFrame of features with the same index as df
    """
    records = []
    total   = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        code   = _ensure_str(row.get(code_col, ""))
        prompt = _ensure_str(row.get(prompt_col, ""))
        records.append(extract_features(code, prompt))
        if show_progress and (i + 1) % 5000 == 0:
            print(f"  [{i+1:,}/{total:,}] features extracted ...")
    return pd.DataFrame(records, index=df.index)