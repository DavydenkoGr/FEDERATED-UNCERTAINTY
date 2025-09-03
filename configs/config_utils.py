# === config_dsl.py ===
# Turn compact strings into your config dicts.
# Grammar (primary forms):
#   risk_<bayes|total|excess>_<logscore|brierscore|spherical|zero-one>_<gt_approx>[_<pred_approx>][_key=value ...]
#   mahalanobis[_key=value ...]
#   gmm[_key=value ...]
#
# Extras:
#   - Brace expansion: risk_bayes_{logscore,brierscore,spherical}_outer
#   - Approximations: outer|1, inner|2, central|3
#   - Synonyms: logscore|log|entropy, brierscore|brier, spherical|sph, zero-one|zeroone|zo
#   - label="My name" overrides auto print_name
#   - Arbitrary key=value pairs pass-through into kwargs (e.g., T=2.0)

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Iterable, Optional
from collections.abc import Iterable as ColIterable
import re
import ast

# Import your enums
# from mdu.unc.constants import UncertaintyType
# from mdu.unc.risk_metrics.constants import GName, RiskType, ApproximationType

# ---- MAPPINGS (edit in one place) -------------------------------------------

TYPE_MAP = {
    "risk": "RISK",          # UncertaintyType.RISK
    "mahalanobis": "MAHALANOBIS",
    "gmm": "GMM",
}

RISK_TYPE_MAP = {
    "bayes": "BAYES_RISK",   # RiskType.BAYES_RISK
    "b": "BAYES_RISK",
    "total": "TOTAL_RISK",
    "tot": "TOTAL_RISK",
    "excess": "EXCESS_RISK",
    "exc": "EXCESS_RISK",
}

GNAME_MAP = {
    # log-score family
    "logscore": "LOG_SCORE", "log": "LOG_SCORE", "entropy": "LOG_SCORE",
    # brier
    "brierscore": "BRIER_SCORE", "brier": "BRIER_SCORE",
    # spherical
    "spherical": "SPHERICAL_SCORE", "sph": "SPHERICAL_SCORE", "sphericalscore": "SPHERICAL_SCORE",
    # zero-one
    "zero-one": "ZERO_ONE_SCORE", "zeroone": "ZERO_ONE_SCORE", "zo": "ZERO_ONE_SCORE", "0-1": "ZERO_ONE_SCORE",
}

APPROX_MAP = {
    "outer": "OUTER", "1": "OUTER",
    "inner": "INNER", "2": "INNER",
    "central": "CENTRAL", "3": "CENTRAL",
}

# Short tags for pretty labels
RISK_SHORT = {"BAYES_RISK": "B", "TOTAL_RISK": "TOT", "EXCESS_RISK": "EXC"}
G_SHORT = {"LOG_SCORE": "log", "BRIER_SCORE": "brier", "SPHERICAL_SCORE": "sph", "ZERO_ONE_SCORE": "zero one"}
APPROX_NUM = {"OUTER": 1, "INNER": 2, "CENTRAL": 3}

# ---- UTILITIES --------------------------------------------------------------

def _expand_braces(s: str) -> List[str]:
    """Expand first {...} occurrence recursively: 'a_{x,y}_b' -> ['a_x_b', 'a_y_b']"""
    m = re.search(r"\{([^{}]+)\}", s)
    if not m:
        return [s]
    start, end = m.span()
    options = [opt.strip() for opt in m.group(1).split(",")]
    out: List[str] = []
    for opt in options:
        out.extend(_expand_braces(s[:start] + opt + s[end:]))
    return out

def _parse_kv(tok: str) -> Optional[Tuple[str, Any]]:
    """Parse key=value; value supports int/float/bool/str (quoted) via literal_eval fallback to raw str."""
    if "=" not in tok:
        return None
    k, v = tok.split("=", 1)
    k = k.strip().lower()
    v = v.strip()
    try:
        val = ast.literal_eval(v)
    except Exception:
        val = v
    return k, val

def _need_pred_approx(risk_type_str: str) -> bool:
    return risk_type_str in ("TOTAL_RISK", "EXCESS_RISK")

def _label_for(cfg: Dict[str, Any], *, default_T: float = 1.0) -> str:
    """Generate print_name if not provided."""
    t = cfg["type"]
    if t == "RISK":
        g = cfg["kwargs"]["g_name"]
        r = cfg["kwargs"]["risk_type"]
        gt = cfg["kwargs"].get("gt_approx")
        pa = cfg["kwargs"].get("pred_approx")
        g_short = G_SHORT.get(g, g.lower())

        if r == "BAYES_RISK":
            # e.g., "B 1 (log)"
            return f"{RISK_SHORT[r]} {APPROX_NUM[gt]} ({g_short})"
        else:
            # e.g., "TOT 1 1 (brier)"
            a = f"{APPROX_NUM[gt]} {APPROX_NUM[pa]}" if pa else f"{APPROX_NUM[gt]}"
            return f"{RISK_SHORT[r]} {a} ({g_short})"
    elif t == "MAHALANOBIS":
        return "Mahalanobis score"
    elif t == "GMM":
        return "GMM score"
    return "measure"

def _err(msg: str, *, spec: str) -> RuntimeError:
    return RuntimeError(f"[config_dsl] {msg}. In spec: '{spec}'")

# ---- CORE PARSER ------------------------------------------------------------

def parse_spec(
    spec: str,
    *,
    default_T: float = 1.0,
    allow_extra_kwargs: bool = True,
) -> Dict[str, Any]:
    """
    Convert one expanded spec (no braces) into a config dict.
    Examples:
      risk_bayes_logscore_outer
      risk_total_brier_outer_inner_T=2.0
      mahalanobis
      gmm_label="My GMM"
    """
    raw = spec.strip()
    if not raw:
        raise _err("Empty spec", spec=spec)

    tokens = [t for t in raw.lower().split("_") if t != ""]
    if not tokens:
        raise _err("No tokens", spec=spec)

    head = tokens.pop(0)
    if head not in TYPE_MAP:
        raise _err(f"Unknown type '{head}'. Allowed: {sorted(TYPE_MAP.keys())}", spec=spec)

    utype = TYPE_MAP[head]

    label_override: Optional[str] = None
    kwargs: Dict[str, Any] = {}

    if utype == "RISK":
        if not tokens:
            raise _err("Risk spec must include risk type", spec=spec)
        r_tok = tokens.pop(0)
        if r_tok not in RISK_TYPE_MAP:
            raise _err(f"Unknown risk type '{r_tok}'. Allowed: {sorted(RISK_TYPE_MAP.keys())}", spec=spec)
        risk_type_str = RISK_TYPE_MAP[r_tok]

        if not tokens:
            raise _err("Risk spec must include scoring rule", spec=spec)
        g_tok = tokens.pop(0)
        if g_tok not in GNAME_MAP:
            raise _err(f"Unknown scoring rule '{g_tok}'. Allowed: {sorted(GNAME_MAP.keys())}", spec=spec)
        g_name_str = GNAME_MAP[g_tok]

        # Approximations
        if not tokens:
            raise _err("Risk spec must include gt_approx", spec=spec)
        gt_tok = tokens.pop(0)
        if gt_tok not in APPROX_MAP:
            raise _err(f"Unknown gt_approx '{gt_tok}'. Allowed: {sorted(APPROX_MAP.keys())}", spec=spec)
        gt_approx_str = APPROX_MAP[gt_tok]

        pred_approx_str: Optional[str] = None
        if _need_pred_approx(risk_type_str):
            if not tokens:
                raise _err("TOTAL/EXCESS must include pred_approx", spec=spec)
            pa_tok = tokens.pop(0)
            if pa_tok not in APPROX_MAP:
                raise _err(f"Unknown pred_approx '{pa_tok}'. Allowed: {sorted(APPROX_MAP.keys())}", spec=spec)
            pred_approx_str = APPROX_MAP[pa_tok]

        # Collect key=value extras
        for tok in tokens:
            kv = _parse_kv(tok)
            if not kv:
                raise _err(f"Unexpected token '{tok}'. Use key=value or remove.", spec=spec)
            k, v = kv
            if k == "label":
                label_override = str(v)
            else:
                kwargs[k] = v

        # Compose kwargs with defaults and required fields
        risk_kwargs = dict(
            g_name=g_name_str,
            risk_type=risk_type_str,
            gt_approx=gt_approx_str,
            T=kwargs.pop("t", default_T),  # allow T override
        )
        if pred_approx_str is not None:
            risk_kwargs["pred_approx"] = pred_approx_str

        # Merge remaining extras if allowed
        if allow_extra_kwargs:
            risk_kwargs.update(kwargs)

        cfg = {
            "type": "RISK",  # UncertaintyType.RISK
            "print_name": label_override or "",  # fill below if empty
            "kwargs": risk_kwargs,
        }

        if not cfg["print_name"]:
            cfg["print_name"] = _label_for(cfg, default_T=default_T)
        return cfg

    # Non-risk types: consume only key=value pairs
    for tok in tokens:
        kv = _parse_kv(tok)
        if not kv:
            raise _err(f"Unexpected token '{tok}'. Use key=value or remove.", spec=spec)
        k, v = kv
        if k == "label":
            label_override = str(v)
        else:
            kwargs[k] = v

    cfg = {
        "type": utype,
        "print_name": label_override or _label_for({"type": utype, "kwargs": {}}, default_T=default_T),
        "kwargs": kwargs if allow_extra_kwargs else {},
    }
    return cfg

# ---- PUBLIC API -------------------------------------------------------------

def build_configs(
    specs: Iterable[str],
    *,
    default_T: float = 1.0,
    allow_extra_kwargs: bool = True,
) -> List[Dict[str, Any]]:
    """
    Expand braces, parse each variant, return a flat list of config dicts.
    """
    out: List[Dict[str, Any]] = []
    for s in specs:
        for expanded in _expand_braces(s):
            out.append(parse_spec(expanded, default_T=default_T, allow_extra_kwargs=allow_extra_kwargs))
    return out

# Optional: a lightweight registry like you wanted
INTERESTING_COMPOSITIONS: Dict[str, List[Dict[str, Any]]] = {}


def _is_config(obj: Any) -> bool:
    return isinstance(obj, dict) and "type" in obj and "kwargs" in obj

def _normalize_to_configs(
    items: Iterable[Any],
    *,
    default_T: float = 1.0,
) -> List[Dict[str, Any]]:
    """Left-to-right flatten:
       - str  -> parse_spec(str)
       - config dict -> pass through
       - list/tuple/set -> recurse
    """
    out: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            out.append(parse_spec(item, default_T=default_T))
        elif _is_config(item):
            out.append(item)  # already a config dict
        elif isinstance(item, ColIterable):
            out.extend(_normalize_to_configs(item, default_T=default_T))
        else:
            raise TypeError(f"Unsupported item in composition: {type(item)}")
    return out

def register_composition(
    name: str,
    *items: Any,
    default_T: float = 1.0,
) -> None:
    """
    Example valid calls:
      register_composition("X", "risk_bayes_logscore_outer")
      register_composition("X", ["risk_bayes_logscore_outer", "risk_excess_brier_outer_outer"])
      register_composition("X", build_configs([...]))  # list of dicts
      register_composition("X", build_configs([...]), "risk_total_logscore_inner_outer")
    """
    INTERESTING_COMPOSITIONS[name] = _normalize_to_configs(items, default_T=default_T)
