"""Function-level coarse compression for masked Python API recommendation.

This module deliberately stops before fine-grained compression.  It reuses
``CodeCompressor.get_condition_ppl`` for AMI (the LongCodeZip PPL-difference
signal), then combines it with a single-file AST data-flow signal (ADF-IF).
"""
from __future__ import annotations

import ast
import re
import time
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


MASK_IDENTIFIER = "__API_RECOMMENDATION_MASK__"
PROMPT_CODE_PREFIX = "请仅输出 5 行 API 签名，每行一个，不要输出编号或其他解释性文字。\n"
PROMPT_OUTPUT_SUFFIX = "推荐的 5 个 API 签名："


def extract_code_from_api_prompt(prompt: str) -> Tuple[str, str, str]:
    """Extract the code-prefix context from a ``new_first100.json`` prompt.

    The evaluation records are code-completion prefixes: their code ends at (or
    shortly after) ``[MASK]`` and is wrapped in a Chinese API-recommendation
    instruction.  The wrapper must not participate in AST/data-flow analysis.
    """
    if PROMPT_CODE_PREFIX not in prompt or PROMPT_OUTPUT_SUFFIX not in prompt:
        raise ValueError("Prompt does not use the new_first100 API-recommendation template.")
    instruction, remainder = prompt.split(PROMPT_CODE_PREFIX, 1)
    code, suffix = remainder.rsplit(PROMPT_OUTPUT_SUFFIX, 1)
    if "[MASK]" not in code:
        raise ValueError("Prompt code context does not contain [MASK].")
    return instruction + PROMPT_CODE_PREFIX, code.rstrip(), PROMPT_OUTPUT_SUFFIX + suffix


def build_compressed_api_prompt(prefix: str, compressed_code: str, suffix: str) -> str:
    """Reattach the original API-recommendation wrapper after compression."""
    return f"{prefix}{compressed_code}\n{suffix}"


@dataclass(frozen=True)
class CodeUnit:
    """A complete function or class method, addressed by its source span."""

    unit_id: str
    qualified_name: str
    name: str
    class_name: Optional[str]
    lineno: int
    end_lineno: int
    code: str


class _UnitVisitor(ast.NodeVisitor):
    def __init__(self, source_lines: List[str]) -> None:
        self.source_lines = source_lines
        self.units: List[CodeUnit] = []
        self.class_stack: List[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.name)
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(child)
            elif isinstance(child, ast.ClassDef):
                self.visit(child)
        self.class_stack.pop()

    def _add_function(self, node: ast.AST) -> None:
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        class_name = ".".join(self.class_stack) or None
        qualified = f"{class_name}.{node.name}" if class_name else node.name
        decorators = getattr(node, "decorator_list", [])
        start = min([node.lineno] + [decorator.lineno for decorator in decorators])
        end = node.end_lineno
        self.units.append(CodeUnit(
            unit_id=f"{qualified}:{start}-{end}", qualified_name=qualified,
            name=node.name, class_name=class_name, lineno=start, end_lineno=end,
            code="\n".join(self.source_lines[start - 1:end]),
        ))
        # Nested functions are independent units too; enclosing code remains intact.
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.visit(child)

    visit_FunctionDef = _add_function
    visit_AsyncFunctionDef = _add_function


def _mask_parse_source(code: str) -> str:
    """Turn the dataset's ``[MASK]`` placeholder into an AST-valid identifier."""
    return code.replace("[MASK]", MASK_IDENTIFIER)


def _names_in(node: ast.AST) -> Set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _assigned_names(node: ast.AST) -> Set[str]:
    names: Set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            targets = child.targets if isinstance(child, ast.Assign) else [child.target]
            for target in targets:
                names.update(_names_in(target))
    return names


def _called_names(node: ast.AST) -> Set[str]:
    names: Set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


class APIRecommendationCoarseCompressor:
    """Reusable AMI + ADF-IF coarse compressor for one Python source file.

    ``scorer`` must expose ``get_condition_ppl(text, question,
    condition_in_question='prefix')`` and ``get_token_length(text)``.  A normal
    :class:`code_compressor.CodeCompressor` is therefore accepted directly and
    its model/tokenizer caches are reused across budgets and repeated calls.
    """

    def __init__(self, scorer: Any, rrf_k: int = 60) -> None:
        self.scorer = scorer
        self.rrf_k = rrf_k
        self._analysis_cache: Dict[str, Dict[str, Any]] = {}

    def analyze(self, code: str) -> Dict[str, Any]:
        """Parse once and compute reusable units, AMI, ADF-IF, and rankings."""
        if code in self._analysis_cache:
            return self._analysis_cache[code]
        started = time.perf_counter()
        parse_code = _mask_parse_source(code)
        try:
            tree = ast.parse(parse_code)
        except SyntaxError as exc:
            raise ValueError(f"Python source cannot be parsed after MASK substitution: {exc}") from exc
        source_lines = code.splitlines()
        visitor = _UnitVisitor(source_lines)
        visitor.visit(tree)
        if not visitor.units:
            raise ValueError("No function or class method was found in the Python file.")
        # ``obj.[MASK]`` becomes ``obj.__API_RECOMMENDATION_MASK__``: in an
        # AST that placeholder is an Attribute.attr string, not a Name node.
        mask_nodes = [node for node in ast.walk(tree)
                      if (isinstance(node, ast.Name) and node.id == MASK_IDENTIFIER)
                      or (isinstance(node, ast.Attribute) and node.attr == MASK_IDENTIFIER)]
        if len(mask_nodes) != 1:
            raise ValueError(f"Expected exactly one [MASK], found {len(mask_nodes)}.")
        mask_line = mask_nodes[0].lineno
        containing = [u for u in visitor.units if u.lineno <= mask_line <= u.end_lineno]
        # The smallest containing unit handles nested functions correctly.
        mask_unit = min(containing, key=lambda u: u.end_lineno - u.lineno) if containing else None
        if mask_unit is None:
            raise ValueError("[MASK] must be inside a function or class method.")

        units = visitor.units
        unit_nodes = {u.unit_id: ast.parse(_mask_parse_source(textwrap.dedent(u.code))) for u in units}
        timings = {"ast_parse_and_partition_seconds": time.perf_counter() - started}
        ami_started = time.perf_counter()
        ami: Dict[str, float] = {}
        for unit in units:
            if unit.unit_id == mask_unit.unit_id:
                continue
            # This is the existing LongCodeZip conditional PPL difference: positive
            # values mean this unit reduces PPL of the mask-unit context.
            ami[unit.unit_id] = float(self.scorer.get_condition_ppl(
                unit.code, mask_unit.code, condition_in_question="prefix"))
        timings["ami_seconds"] = time.perf_counter() - ami_started

        adf_started = time.perf_counter()
        adf, required = self._adf_if(units, unit_nodes, mask_unit.unit_id)
        timings["adf_if_seconds"] = time.perf_counter() - adf_started
        rankings = self._rank_and_fuse(units, mask_unit.unit_id, ami, adf)
        timings["ranking_seconds"] = time.perf_counter() - adf_started - timings["adf_if_seconds"]
        analysis = {
            "original_code": code, "units": units, "mask_unit_id": mask_unit.unit_id,
            "required_dependency_ids": sorted(required), "ami": ami, "adf_if": adf,
            "rankings": rankings, "timings": timings,
            "original_tokens": self.scorer.get_token_length(code),
        }
        self._analysis_cache[code] = analysis
        return analysis

    def _adf_if(self, units: List[CodeUnit], unit_nodes: Dict[str, ast.AST], mask_id: str) -> Tuple[Dict[str, float], Set[str]]:
        """A conservative single-file, unit-level data-flow/call relevance score."""
        mask_tree = unit_nodes[mask_id]
        mask_node = next(n for n in ast.walk(mask_tree)
                         if (isinstance(n, ast.Name) and n.id == MASK_IDENTIFIER)
                         or (isinstance(n, ast.Attribute) and n.attr == MASK_IDENTIFIER))
        parent: Dict[ast.AST, ast.AST] = {c: n for n in ast.walk(mask_tree) for c in ast.iter_child_nodes(n)}
        attribute = mask_node if isinstance(mask_node, ast.Attribute) else parent.get(mask_node)
        receiver_names = _names_in(attribute.value) if isinstance(attribute, ast.Attribute) else set()
        mask_calls = _called_names(mask_tree)
        mask_names = _names_in(mask_tree)
        by_name: Dict[str, List[str]] = {}
        for unit in units:
            by_name.setdefault(unit.name, []).append(unit.unit_id)
        required = {candidate for call in mask_calls for candidate in by_name.get(call, []) if candidate != mask_id}
        scores: Dict[str, float] = {}
        for unit in units:
            if unit.unit_id == mask_id:
                continue
            node = unit_nodes[unit.unit_id]
            calls, defined, used = _called_names(node), _assigned_names(node), _names_in(node)
            score = 0.0
            # Direct callee of the MASK unit: return/object source can flow to it.
            if unit.unit_id in required:
                score += 4.0
            # Calls into the mask unit or sibling method/callee coupling.
            if unit.name in mask_calls or calls.intersection(mask_calls):
                score += 1.5
            # Receiver/source propagation.  Assignment, parameters, and returns are
            # all represented by AST names, while return gives a stronger signal.
            shared_receiver = receiver_names.intersection(defined | used)
            if shared_receiver:
                score += 1.0 + 0.5 * len(shared_receiver)
            if any(isinstance(n, ast.Return) for n in ast.walk(node)) and unit.name in mask_calls:
                score += 1.5
            # A function consuming/producing names used around the masked expression.
            shared_context = (defined | used).intersection(mask_names)
            score += min(1.0, 0.15 * len(shared_context))
            scores[unit.unit_id] = score
        return scores, required

    def _rank_and_fuse(self, units: List[CodeUnit], mask_id: str, ami: Dict[str, float], adf: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        candidates = [u.unit_id for u in units if u.unit_id != mask_id]
        ami_order = sorted(candidates, key=lambda u: (-ami[u], u))
        adf_order = sorted(candidates, key=lambda u: (-adf[u], u))
        ami_rank = {u: i + 1 for i, u in enumerate(ami_order)}
        adf_rank = {u: i + 1 for i, u in enumerate(adf_order)}
        return {u: {"ami_rank": ami_rank[u], "adf_if_rank": adf_rank[u],
                    "fused_score": 1 / (self.rrf_k + ami_rank[u]) + 1 / (self.rrf_k + adf_rank[u])}
                for u in candidates}

    def compress(self, code: str, token_budget: int) -> Dict[str, Any]:
        """Select complete units using exact 0/1 knapsack and reconstruct valid Python."""
        if token_budget <= 0:
            raise ValueError("token_budget must be positive.")
        started = time.perf_counter()
        analysis = self.analyze(code)
        units: List[CodeUnit] = analysis["units"]
        forced = {analysis["mask_unit_id"], *analysis["required_dependency_ids"]}
        costs = {u.unit_id: self.scorer.get_token_length(u.code) for u in units}
        forced_cost = sum(costs[u] for u in forced)
        candidates = [u for u in units if u.unit_id not in forced]
        remaining = max(0, token_budget - forced_cost)
        selected = set(forced) | self._knapsack(candidates, costs, analysis["rankings"], remaining)
        compressed = self._reconstruct(code, selected, units)
        try:
            ast.parse(_mask_parse_source(compressed))
            syntax = {"valid": True, "error": None}
        except SyntaxError as exc:  # defensive: result is returned with explicit status
            syntax = {"valid": False, "error": str(exc)}
        tokens_after = self.scorer.get_token_length(compressed)
        unit_details = []
        for u in units:
            data = {"unit_id": u.unit_id, "qualified_name": u.qualified_name,
                    "line_range": [u.lineno, u.end_lineno], "tokens": costs[u.unit_id],
                    "selected": u.unit_id in selected, "required_dependency": u.unit_id in forced}
            if u.unit_id != analysis["mask_unit_id"]:
                data.update({"ami": analysis["ami"][u.unit_id], "adf_if": analysis["adf_if"][u.unit_id],
                             **analysis["rankings"][u.unit_id]})
            unit_details.append(data)
        timings = dict(analysis["timings"])
        timings["selection_and_reconstruction_seconds"] = time.perf_counter() - started
        return {"original_code": code, "compressed_code": compressed,
                "mask_unit": analysis["mask_unit_id"], "required_dependency_units": sorted(forced - {analysis["mask_unit_id"]}),
                "candidate_units": unit_details, "selected_units": [u.unit_id for u in units if u.unit_id in selected],
                "deleted_units": [u.unit_id for u in units if u.unit_id not in selected],
                "original_tokens": analysis["original_tokens"], "compressed_tokens": tokens_after,
                "compression_ratio": tokens_after / analysis["original_tokens"] if analysis["original_tokens"] else 1.0,
                "token_budget": token_budget, "mandatory_tokens": forced_cost,
                "budget_exceeded_by_mandatory_units": max(0, forced_cost - token_budget),
                "syntax_check": syntax, "timings": timings}

    def compress_dataset(self, records: List[Dict[str, Any]], token_budget: int) -> Dict[str, Any]:
        """Compress valid API-recommendation records from ``new_first100.json``.

        A dataset record is the unit of work, not a standalone Python file.  Bad
        or Python-2/incomplete code prefixes are reported per record and do not
        abort a batch experiment; valid records reuse this object's model and
        AMI/static-analysis caches.
        """
        started = time.perf_counter()
        results: List[Dict[str, Any]] = []
        succeeded = 0
        for index, record in enumerate(records):
            item: Dict[str, Any] = {"id": record.get("id", index), "gt": record.get("gt"),
                                    "original_prompt": record.get("prompt")}
            try:
                prefix, code, suffix = extract_code_from_api_prompt(record["prompt"])
                coarse = self.compress(code, token_budget)
                item.update({"status": "ok", "compressed_prompt": build_compressed_api_prompt(prefix, coarse["compressed_code"], suffix),
                             "coarse_result": coarse})
                succeeded += 1
            except (KeyError, TypeError, ValueError) as exc:
                item.update({"status": "skipped", "error": str(exc)})
            results.append(item)
        return {"dataset_records": len(records), "compressed_records": succeeded,
                "skipped_records": len(records) - succeeded, "token_budget": token_budget,
                "records": results, "dataset_seconds": time.perf_counter() - started}

    @staticmethod
    def _knapsack(candidates: List[CodeUnit], costs: Dict[str, int], rankings: Dict[str, Dict[str, float]], capacity: int) -> Set[str]:
        # Exact 0/1 DP; coarse unit counts are normally small.  A sparse DP avoids
        # allocating a huge table when a caller gives a very large token budget.
        states: Dict[int, Tuple[float, Set[str]]] = {0: (0.0, set())}
        for unit in candidates:
            weight, value = costs[unit.unit_id], rankings[unit.unit_id]["fused_score"]
            updates = dict(states)
            for used, (score, picked) in states.items():
                if used + weight <= capacity and score + value > updates.get(used + weight, (-1.0, set()))[0]:
                    updates[used + weight] = (score + value, picked | {unit.unit_id})
            states = updates
        return max(states.values(), key=lambda item: item[0])[1]

    @staticmethod
    def _reconstruct(code: str, selected: Set[str], units: List[CodeUnit]) -> str:
        """Replace omitted complete units with syntactically valid, indented ``pass``."""
        lines = code.splitlines()
        # Nested units lie in their parent span; only replace outermost omitted units.
        omitted = [u for u in units if u.unit_id not in selected]
        outermost = []
        for unit in omitted:
            if not any(other.lineno <= unit.lineno and unit.end_lineno <= other.end_lineno and other.unit_id != unit.unit_id for other in omitted):
                outermost.append(unit)
        for unit in sorted(outermost, key=lambda u: u.lineno, reverse=True):
            indent = re.match(r"\s*", lines[unit.lineno - 1]).group(0)
            # The source span includes decorators, so no decorator remains attached
            # to the placeholder after an omitted method/function is replaced.
            replacement = [f"{indent}# [API_COARSE_OMITTED] {unit.qualified_name}", f"{indent}pass"]
            lines[unit.lineno - 1:unit.end_lineno] = replacement
        return "\n".join(lines) + ("\n" if code.endswith("\n") else "")
