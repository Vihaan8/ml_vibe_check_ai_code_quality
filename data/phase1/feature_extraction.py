"""
feature_extraction.py
Phase 1: Static feature extraction for Vibe Check (IDS 705)

Four feature categories per proposal:
  1. Classical software metrics   (LOC, cyclomatic complexity, nesting depth, halstead proxies)
  2. AST structural features      (node-type counts from the parse tree)
  3. Prompt-code alignment        (library coverage, length ratio, function signature match)
  4. LLM smell features           (hardcoded returns, placeholder names, suspiciously short code)

Usage
-----
    from feature_extraction import extract_features, extract_features_batch

    # single sample
    feats = extract_features(code_str, prompt_str)   # -> dict

    # DataFrame of samples
    df_features = extract_features_batch(df, code_col="generated_code", prompt_col="prompt")
"""

import ast
import re
import textwrap
from typing import Optional

import pandas as pd
import numpy as np

from radon.raw import analyze
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit

# ---------------------------------------------------------------------------
# 1. Classical software metrics
# ---------------------------------------------------------------------------


def _safe_parse(code: str) -> Optional[ast.Module]:
    """Try to parse code; return None if syntax error or non-string input."""
    if not isinstance(code, str):
        return None
    try:
        return ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return None


def _radon_raw_metrics(code: str) -> dict:
    """Raw code metrics from radon.raw.analyze()."""
    try:
        raw = analyze(code)
        return {
            "loc_total": raw.loc,
            "loc_blank": raw.blank,
            "loc_comment": getattr(raw, "single_comments", raw.comments),
            "loc_code": raw.sloc,
            "loc_logical": raw.lloc,
            "loc_multi": raw.multi,
        }
    except Exception:
        return {
            "loc_total": 0,
            "loc_blank": 0,
            "loc_comment": 0,
            "loc_code": 0,
            "loc_logical": 0,
            "loc_multi": 0,
        }


def _radon_cc_metrics(code: str) -> dict:
    """Cyclomatic complexity from radon.complexity.cc_visit()."""
    try:
        blocks = cc_visit(code)
        complexities = [b.complexity for b in blocks]

        if complexities:
            cc_max = max(complexities)
            cc_mean = sum(complexities) / len(complexities)
            cc_blocks = len(complexities)
        else:
            cc_max = 1 if code.strip() else 0
            cc_mean = float(cc_max)
            cc_blocks = 0

        return {
            "classical_cyclomatic_complexity": cc_max,
            "classical_cyclomatic_complexity_mean": round(cc_mean, 4),
            "classical_cc_block_count": cc_blocks,
        }
    except Exception:
        return {
            "classical_cyclomatic_complexity": 0,
            "classical_cyclomatic_complexity_mean": 0.0,
            "classical_cc_block_count": 0,
        }


def _halstead_metrics(code: str) -> dict:
    """Halstead metrics from radon.metrics.h_visit()."""
    try:
        h = h_visit(code).total
        return {
            "halstead_unique_operators": h.h1,
            "halstead_unique_operands": h.h2,
            "halstead_total_operators": h.N1,
            "halstead_total_operands": h.N2,
            "halstead_vocabulary": h.h,
            "halstead_length": h.N,
            "halstead_volume": round(h.volume, 2),
            "halstead_difficulty": round(h.difficulty, 2),
            "halstead_effort": round(h.effort, 2),
            "halstead_time": round(h.time, 2),
            "halstead_bugs": round(h.bugs, 4),
        }
    except Exception:
        return {
            "halstead_unique_operators": 0,
            "halstead_unique_operands": 0,
            "halstead_total_operators": 0,
            "halstead_total_operands": 0,
            "halstead_vocabulary": 0,
            "halstead_length": 0,
            "halstead_volume": 0.0,
            "halstead_difficulty": 0.0,
            "halstead_effort": 0.0,
            "halstead_time": 0.0,
            "halstead_bugs": 0.0,
        }


def _maintainability_index(code: str) -> dict:
    """Optional: MI from radon.metrics.mi_visit()."""
    try:
        mi = mi_visit(code, multi=True)
        return {
            "classical_maintainability_index": round(mi, 2),
        }
    except Exception:
        return {
            "classical_maintainability_index": 0.0,
        }


def _max_nesting_depth(tree: Optional[ast.Module]) -> int:
    """Maximum control-flow nesting depth."""
    if tree is None:
        return 0

    nesting_types = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.ExceptHandler)

    def _depth(node, current=0):
        if isinstance(node, nesting_types):
            current += 1
        return max(
            [current] + [_depth(child, current) for child in ast.iter_child_nodes(node)]
        )

    return _depth(tree)


# ---------------------------------------------------------------------------
# 2. AST structural features
# ---------------------------------------------------------------------------

_AST_NODE_TYPES = {
    # control flow
    "ast_if_count": ast.If,
    "ast_for_count": ast.For,
    "ast_while_count": ast.While,
    "ast_try_count": ast.Try,
    "ast_except_count": ast.ExceptHandler,
    "ast_with_count": ast.With,
    # functions / classes
    "ast_funcdef_count": ast.FunctionDef,
    "ast_asyncfuncdef_count": ast.AsyncFunctionDef,
    "ast_classdef_count": ast.ClassDef,
    # returns / yields
    "ast_return_count": ast.Return,
    "ast_yield_count": ast.Yield,
    # comprehensions
    "ast_listcomp_count": ast.ListComp,
    "ast_dictcomp_count": ast.DictComp,
    "ast_setcomp_count": ast.SetComp,
    "ast_genexpr_count": ast.GeneratorExp,
    # imports
    "ast_import_count": ast.Import,
    "ast_importfrom_count": ast.ImportFrom,
    # assignments / augmented
    "ast_assign_count": ast.Assign,
    "ast_augassign_count": ast.AugAssign,
    # assertions / raises / deletes
    "ast_assert_count": ast.Assert,
    "ast_raise_count": ast.Raise,
    "ast_delete_count": ast.Delete,
    # lambdas
    "ast_lambda_count": ast.Lambda,
    # pass / ellipsis (stub indicators)
    "ast_pass_count": ast.Pass,
    "ast_ellipsis_count": ast.Constant,  # filtered below
}


def _ast_node_counts(tree: Optional[ast.Module]) -> dict:
    counts = {k: 0 for k in _AST_NODE_TYPES}
    if tree is None:
        return counts
    for node in ast.walk(tree):
        for feat, node_type in _AST_NODE_TYPES.items():
            if isinstance(node, node_type):
                # Ellipsis: only count `...` constants
                if feat == "ast_ellipsis_count":
                    if isinstance(getattr(node, "value", None), type(...)):
                        counts[feat] += 1
                else:
                    counts[feat] += 1
    return counts


def _has_error_handling(tree: Optional[ast.Module]) -> int:
    """1 if code has at least one try/except block."""
    if tree is None:
        return 0
    return int(any(isinstance(n, ast.Try) for n in ast.walk(tree)))


def _function_arg_count(tree: Optional[ast.Module]) -> int:
    """Total number of arguments across all function definitions."""
    if tree is None:
        return 0
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            count += len(node.args.args)
    return count


def _unique_variable_names(tree: Optional[ast.Module]) -> int:
    """Count of unique Name nodes used in assignments."""
    if tree is None:
        return 0
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return len(names)


# ---------------------------------------------------------------------------
# 3. Prompt-code alignment features
# ---------------------------------------------------------------------------

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
    "typing": ["typing"],
}


def _extract_imported_libraries(tree: Optional[ast.Module]) -> set:
    """Return set of top-level module names imported in the code."""
    if tree is None:
        return set()
    libs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                libs.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                libs.add(node.module.split(".")[0])
    return libs


def _prompt_library_mentions(prompt: str) -> set:
    """
    Heuristically extract library names mentioned in the prompt.
    Looks for known library names and their aliases.
    """
    prompt_lower = prompt.lower()
    mentioned = set()
    for lib, aliases in _LIB_ALIASES.items():
        if any(a in prompt_lower for a in aliases):
            mentioned.add(lib)
    return mentioned


def _prompt_alignment_features(
    code: str, prompt: str, tree: Optional[ast.Module]
) -> dict:
    imported = _extract_imported_libraries(tree)
    prompt_libs = _prompt_library_mentions(prompt)

    # Library coverage: fraction of prompt-mentioned libs that appear in imports
    if prompt_libs:
        lib_coverage = len(imported & prompt_libs) / len(prompt_libs)
    else:
        lib_coverage = 1.0  # no libs mentioned → not penalized

    # Missing libraries: count of prompt-mentioned libs NOT imported
    missing_libs = len(prompt_libs - imported)

    # Code-to-prompt length ratio (chars)
    prompt_len = max(len(prompt), 1)
    code_len = len(code)
    length_ratio = code_len / prompt_len

    # Prompt mentions return type / output structure
    has_return_hint = int(
        any(kw in prompt.lower() for kw in ["return", "returns", "output", "result"])
    )

    # Does the generated function name appear to match the prompt?
    func_name_match = 0
    if tree is not None:
        func_names = [
            node.name.lower()
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        prompt_words = set(re.findall(r"\b\w+\b", prompt.lower()))
        func_name_match = int(any(name in prompt_words for name in func_names))

    # Word overlap between prompt and code identifiers
    code_identifiers = set(re.findall(r"\b[a-zA-Z_]\w*\b", code))
    prompt_words_set = set(re.findall(r"\b[a-zA-Z_]\w*\b", prompt.lower()))
    identifier_overlap = (
        len(code_identifiers & prompt_words_set) / len(prompt_words_set)
        if prompt_words_set
        else 0
    )

    return {
        "align_lib_coverage": round(lib_coverage, 4),
        "align_missing_libs": missing_libs,
        "align_code_prompt_length_ratio": round(length_ratio, 4),
        "align_has_return_hint": has_return_hint,
        "align_func_name_matches_prompt": func_name_match,
        "align_identifier_overlap": round(identifier_overlap, 4),
        "align_num_imports": len(imported),
        "align_num_prompt_libs": len(prompt_libs),
    }


# ---------------------------------------------------------------------------
# 4. LLM smell features
# ---------------------------------------------------------------------------

# Placeholder / stub patterns common in LLM cop-outs
_PLACEHOLDER_PATTERNS = [
    r"\bpass\b",
    r"\.\.\.",
    r"#\s*TODO",
    r"#\s*FIXME",
    r"#\s*PLACEHOLDER",
    r"raise\s+NotImplementedError",
    r"return\s+None\s*$",  # bare return None with nothing else
    r"your\s+code\s+here",
    r"implement\s+this",
]

# Generic / meaningless variable names
_GENERIC_NAMES = {
    "result",
    "output",
    "data",
    "res",
    "ret",
    "temp",
    "tmp",
    "val",
    "x",
    "y",
    "z",
    "foo",
    "bar",
}


def _llm_smell_features(code: str, tree: Optional[ast.Module]) -> dict:
    lines = code.splitlines()
    code_lower = code.lower()

    # ---- Hardcoded return smell ----
    # A function that only contains a single `return <literal>` and nothing else
    hardcoded_return_funcs = 0
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body_stmts = [n for n in node.body if not isinstance(n, ast.Expr)]
                if len(body_stmts) == 1 and isinstance(body_stmts[0], ast.Return):
                    ret_val = body_stmts[0].value
                    if isinstance(ret_val, ast.Constant):
                        hardcoded_return_funcs += 1

    # ---- Placeholder / stub patterns ----
    placeholder_hits = sum(
        len(re.findall(pat, code, re.IGNORECASE | re.MULTILINE))
        for pat in _PLACEHOLDER_PATTERNS
    )

    # ---- Generic variable name ratio ----
    if tree is not None:
        assigned_names = [
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        ]
        generic_ratio = (
            sum(1 for n in assigned_names if n in _GENERIC_NAMES) / len(assigned_names)
            if assigned_names
            else 0
        )
        n_assigned = len(assigned_names)
    else:
        generic_ratio = 0
        n_assigned = 0

    # ---- Immediate return (function returns on line 2 of body, no logic) ----
    immediate_return = 0
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                non_docstring_body = [
                    n
                    for n in node.body
                    if not (
                        isinstance(n, ast.Expr)
                        and isinstance(getattr(n, "value", None), ast.Constant)
                    )
                ]
                if non_docstring_body and isinstance(non_docstring_body[0], ast.Return):
                    immediate_return += 1

    # ---- Comment density ----
    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
    comment_density = comment_lines / max(len(lines), 1)

    # ---- Magic numbers ----
    magic_numbers = 0
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if node.value not in (0, 1, -1, 2, True, False):
                    magic_numbers += 1

    # ---- String literal used as entire return (e.g., return "not implemented") ----
    string_return_count = len(re.findall(r"return\s+['\"]", code, re.IGNORECASE))

    # ---- Suspicious shortness ----
    # (absolute; relative-to-task handled at dataset level in Phase 2)
    is_very_short = int(len([l for l in lines if l.strip()]) <= 5)

    # ---- No imports at all ----
    has_any_import = int(bool(re.search(r"^\s*(import|from)\s+", code, re.MULTILINE)))

    # ---- Duplicate lines ratio ----
    non_blank_lines = [l.strip() for l in lines if l.strip()]
    if non_blank_lines:
        dup_ratio = 1 - len(set(non_blank_lines)) / len(non_blank_lines)
    else:
        dup_ratio = 0

    return {
        "smell_hardcoded_return_funcs": hardcoded_return_funcs,
        "smell_placeholder_hits": placeholder_hits,
        "smell_generic_var_ratio": round(generic_ratio, 4),
        "smell_n_assigned_vars": n_assigned,
        "smell_immediate_return_funcs": immediate_return,
        "smell_comment_density": round(comment_density, 4),
        "smell_magic_numbers": magic_numbers,
        "smell_string_return_count": string_return_count,
        "smell_is_very_short": is_very_short,
        "smell_has_any_import": has_any_import,
        "smell_duplicate_line_ratio": round(dup_ratio, 4),
    }


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

PARSE_ERROR_SENTINEL = -1  # used for classical metrics when code is unparseable


def extract_features(code: str, prompt: str = "") -> dict:
    """
    Extract all Phase 1 features for a single (code, prompt) pair.
    """
    tree = _safe_parse(code)
    parse_error = int(tree is None)

    feats: dict = {}

    # -- meta --
    feats["meta_parse_error"] = parse_error
    feats["meta_code_len_chars"] = len(code)

    # -- 1. Classical --
    feats.update(_radon_raw_metrics(code))
    feats.update(_radon_cc_metrics(code))
    feats["classical_max_nesting_depth"] = _max_nesting_depth(tree)
    feats["classical_func_arg_count"] = _function_arg_count(tree)
    feats["classical_unique_var_count"] = _unique_variable_names(tree)
    feats["classical_has_error_handling"] = _has_error_handling(tree)
    feats.update(_halstead_metrics(code))
    feats.update(_maintainability_index(code))

    # -- 2. AST structural --
    feats.update(_ast_node_counts(tree))

    # -- 3. Prompt-code alignment --
    feats.update(_prompt_alignment_features(code, prompt, tree))

    # -- 4. LLM smells --
    feats.update(_llm_smell_features(code, tree))

    return feats


def extract_features_batch(
    df: pd.DataFrame,
    code_col: str = "generated_code",
    prompt_col: Optional[str] = "prompt",
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Extract features for every row in a DataFrame.

    Parameters
    ----------
    df         : input DataFrame; must have `code_col`; `prompt_col` optional
    code_col   : column containing source code strings
    prompt_col : column containing task prompt strings (pass None to skip alignment feats)
    show_progress : print a simple progress line every 5000 rows

    Returns
    -------
    DataFrame of features aligned with df's index
    """
    records = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        code = row.get(code_col, "")
        code = "" if not isinstance(code, str) else code
        prompt = row.get(prompt_col, "") if prompt_col else ""
        prompt = "" if not isinstance(prompt, str) else prompt
        records.append(extract_features(code, prompt))
        if show_progress and (i + 1) % 5000 == 0:
            print(f"  [{i+1}/{total}] features extracted …")
    return pd.DataFrame(records, index=df.index)
