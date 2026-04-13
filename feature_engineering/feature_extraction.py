"""
feature_extraction.py

Static feature extraction for Vibe Check. Four feature groups:
    1. Classical software metrics (LOC, cyclomatic complexity, nesting depth)
    2. AST structural features (control flow, error handling, import counts)
    3. Prompt-code alignment (library coverage, missing libs, length ratio)
    4. LLM smell features (hardcoded returns, placeholders, suspiciously short)
"""

import ast
import re
import textwrap
from typing import Optional
import pandas as pd
from radon.complexity import cc_visit
from radon.raw import analyze



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


# 1. Classical software metrics


def _classical_features(code: str, tree: Optional[ast.Module]) -> dict:
    """
    LOC        : total physical lines (via radon.raw.analyze)
    CC         : cyclomatic complexity summed across all functions (via radon cc_visit)
    Nesting    : maximum control-flow nesting depth (via ast)
    """
    # LOC by randon
    loc = 0
    if code.strip():
        try:
            loc = analyze(code).sloc   # source lines of code (non-blank, non-comment)
        except SyntaxError:
            loc = len([l for l in code.splitlines() if l.strip()])

    # Cyclomatic complexity by radon
    cc = 0
    if code.strip():
        try:
            blocks = cc_visit(code)
            cc = sum(b.complexity for b in blocks) if blocks else 1
        except Exception:
            cc = 0

    # Max nesting depth by ast
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



# 2. AST structural features


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



# 3. Prompt-code alignment features


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


def _parse_libs_field(libs_str: str) -> set:
    """Parse the libs column from BigCodeBench (e.g. "['pandas', 'requests']")."""
    if not isinstance(libs_str, str) or not libs_str.strip():
        return set()
    try:
        parsed = ast.literal_eval(libs_str)
        return set(parsed) if isinstance(parsed, list) else set()
    except (ValueError, SyntaxError):
        return set()


def _alignment_features(code: str, prompt: str, tree: Optional[ast.Module],
                         required_libs: set) -> dict:
    """
    lib_coverage  : fraction of required libraries actually imported
    missing_libs  : count of required libraries NOT imported
    length_ratio  : len(code) / len(prompt)
    """
    imported = _imported_libs(tree)

    lib_coverage = (
        len(imported & required_libs) / len(required_libs) if required_libs else 1.0
    )

    return {
        "align_lib_coverage":   round(lib_coverage, 4),
        "align_missing_libs":   len(required_libs - imported),
        "align_length_ratio":   round(len(code) / max(len(prompt), 1), 4),
    }


# 4. LLM smell features

def _smell_features(code: str, tree: Optional[ast.Module]) -> dict:
    hardcoded_returns = 0
    placeholder_hits = 0

    if tree is not None:
        for node in ast.walk(tree):
            # Hardcoded return: function body is just `return <constant>`
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                real = [n for n in node.body if not isinstance(n, ast.Expr)]
                if (len(real) == 1
                        and isinstance(real[0], ast.Return)
                        and isinstance(real[0].value, ast.Constant)):
                    hardcoded_returns += 1

            # Placeholder: pass statements
            if isinstance(node, ast.Pass):
                placeholder_hits += 1

            # Placeholder: bare Ellipsis (...)
            if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
                if node.value.value is Ellipsis:
                    placeholder_hits += 1

            # Placeholder: raise NotImplementedError
            if isinstance(node, ast.Raise) and node.exc is not None:
                if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
                    if node.exc.func.id == "NotImplementedError":
                        placeholder_hits += 1

    # TODO/FIXME in string literals or comments
    placeholder_hits += len(re.findall(r'\b(TODO|FIXME|HACK|XXX)\b', code, re.IGNORECASE))

    non_blank = [l for l in code.splitlines() if l.strip()]
    is_very_short = int(len(non_blank) <= 5)

    return {
        "smell_hardcoded_return_funcs": hardcoded_returns,
        "smell_placeholder_hits":       placeholder_hits,
        "smell_is_very_short":          is_very_short,
    }


# Main extraction functions

def extract_features(code: str, prompt: str = "", libs: str = "") -> dict:
    """Extract all features for a single (code, prompt, libs) sample."""
    code   = _ensure_str(code)
    prompt = _ensure_str(prompt)
    tree   = _safe_parse(code)
    required_libs = _parse_libs_field(libs)

    feats = {"meta_parse_error": int(tree is None)}
    feats.update(_classical_features(code, tree))
    feats.update(_ast_features(tree))
    feats.update(_alignment_features(code, prompt, tree, required_libs))
    feats.update(_smell_features(code, tree))
    return feats


def extract_features_batch(
    df: pd.DataFrame,
    code_col: str = "generated_code",
    prompt_col: str = "prompt",
    libs_col: str = "libs",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Extract features for every row in a DataFrame."""
    records = []
    total   = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        code   = _ensure_str(row.get(code_col, ""))
        prompt = _ensure_str(row.get(prompt_col, ""))
        libs   = _ensure_str(row.get(libs_col, ""))
        records.append(extract_features(code, prompt, libs))
        if show_progress and (i + 1) % 5000 == 0:
            print(f"  [{i+1:,}/{total:,}] features extracted ...")
    return pd.DataFrame(records, index=df.index)