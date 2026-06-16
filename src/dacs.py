"""
DACS — Diversity-Aware Configuration Selector.

Lightweight enhancement module for EvoTest's Evolver-to-Actor configuration
selection step. It sits between Evolver candidate generation and the next
Actor episode, re-ranking and filtering candidate actor configurations
based on:

  - relevance to the latest trajectory's progress and failure cues,
  - diversity against already-selected candidates (relevance-weighted
    DPP-style greedy selection),
  - failure risk derived from cross-episode failure memory and trajectory
    loops.

The module is intentionally dependency-free (no embedding model, no extra
heavyweight library) and degrades gracefully — any internal exception
returns ``fallback=True`` so the caller can fall back to the original UCB
selection logic without crashing the experiment.

It does not hardcode any game name, room, item, prompt or API key.
"""

import re
import traceback
from collections import Counter
from typing import Any, Dict, List, Optional


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]+")

_STOPWORDS = set(
    """the a an and or of to in is are was were be been being have has had do does did
    this that these those it its on at for with by as not no so if then else than
    i you he she we they me him her us them my your our their his hers theirs
    will would can could should may might must shall up down out over under again
    """.split()
)


def _tokenize(text: Any) -> List[str]:
    if text is None:
        return []
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return []
    return [
        t.lower()
        for t in _TOKEN_RE.findall(text)
        if t.lower() not in _STOPWORDS and len(t) > 1
    ]


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0


def _bounded(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _to_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return " ".join(_to_text(v) for v in x)
    if isinstance(x, dict):
        parts = []
        for k, v in x.items():
            parts.append(str(k))
            parts.append(_to_text(v))
        return " ".join(parts)
    try:
        return str(x)
    except Exception:
        return ""


def build_config_profile(cfg: Any) -> Dict[str, Any]:
    """Build a robust profile from a candidate actor configuration.

    Tolerates dicts, strings, tuples and arbitrary objects. Missing fields
    become empty strings. The original config is preserved on the ``raw``
    key so callers can still pass it back unchanged.
    """
    profile: Dict[str, Any] = {
        "raw": cfg,
        "prompt": "",
        "code": "",
        "memory": "",
        "tools": "",
        "hyperparams": "",
        "success_memory": "",
        "failure_memory": "",
        "history_text": "",
        "score": None,
        "depth": 0,
    }
    if cfg is None:
        return profile
    if isinstance(cfg, str):
        profile["prompt"] = cfg
        return profile
    if isinstance(cfg, (list, tuple)):
        if len(cfg) > 0:
            profile["prompt"] = _to_text(cfg[0])
        if len(cfg) > 1:
            profile["code"] = _to_text(cfg[1])
        return profile
    if isinstance(cfg, dict):
        field_map = [
            ("prompt", ["prompt", "guiding_prompt", "instruction", "system_prompt"]),
            ("code", ["code", "state_extractor", "tool_code"]),
            ("memory", ["memory", "mem"]),
            ("tools", ["tools", "tool_use", "tool_routine"]),
            ("hyperparams", ["hyperparams", "params", "config"]),
            ("success_memory", ["success_memory", "positives"]),
            ("failure_memory", ["failure_memory", "negatives"]),
            ("history_text", ["game_history", "history", "transcript"]),
        ]
        for target, candidates in field_map:
            for c in candidates:
                if c in cfg and cfg[c] is not None:
                    profile[target] = _to_text(cfg[c])
                    break
        if "score" in cfg:
            profile["score"] = cfg.get("score")
        if "depth" in cfg:
            try:
                profile["depth"] = int(cfg.get("depth", 0))
            except Exception:
                profile["depth"] = 0
        return profile
    for attr in ("prompt", "code", "memory", "score", "depth"):
        if hasattr(cfg, attr):
            try:
                v = getattr(cfg, attr)
                profile[attr] = _to_text(v) if attr not in ("score", "depth") else v
            except Exception:
                pass
    return profile


def _profile_tokens(profile: Dict[str, Any]) -> List[str]:
    return _tokenize(
        " ".join(
            [
                profile.get("prompt", ""),
                profile.get("code", ""),
                profile.get("memory", ""),
                profile.get("tools", ""),
                profile.get("hyperparams", ""),
                profile.get("success_memory", ""),
                profile.get("failure_memory", ""),
            ]
        )
    )


def extract_proxy_states(
    transcript: Any,
    score_delta: Optional[float] = None,
    final_score: Optional[float] = None,
    success_memory: Optional[List[Dict[str, Any]]] = None,
    failure_memory: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[str]]:
    """Extract proxy-state tokens that approximate task-relevant signals.

    Uses generic text heuristics only — no per-game rules. ``transcript``
    can be a raw string or any object convertible to text.
    """
    proxy = {
        "progress_tokens": [],
        "positive_progress_tokens": [],
        "neutral_progress_tokens": [],
        "invalid_tokens": [],
        "loop_tokens": [],
        "success_tokens": [],
        "failure_tokens": [],
        "failure_pattern_tokens": [],
        "inventory_tokens": [],
        "location_tokens": [],
        "object_tokens": [],
    }
    text = transcript if isinstance(transcript, str) else _to_text(transcript)

    steps = _parse_transcript_steps(text)
    if steps:
        positive_chunks: List[str] = []
        neutral_chunks: List[str] = []
        failure_chunks: List[str] = []
        inventory_chunks: List[str] = []
        observation_heads: List[str] = []
        object_chunks: List[str] = []
        seen_obs = set()
        seen_inv = set()
        last_obs = ""
        last_inv = ""

        for step in steps:
            obs = step.get("obs", "")
            inv = step.get("inv", "")
            action = step.get("action", "")
            reward = _safe_float(step.get("reward"))
            obs_key = _normalize_space(obs)[:260]
            inv_key = _normalize_space(inv)[:260]
            obs_changed = bool(obs_key and obs_key != last_obs)
            inv_changed = bool(inv_key and inv_key != last_inv)
            new_obs = bool(obs_key and obs_key not in seen_obs)
            new_inv = bool(inv_key and inv_key not in seen_inv)

            if obs_key:
                seen_obs.add(obs_key)
                last_obs = obs_key
                observation_heads.append(_first_observation_line(obs))
                object_chunks.extend(_extract_visible_object_phrases(obs))
            if inv_key:
                seen_inv.add(inv_key)
                last_inv = inv_key

            action_text = f"{action} {obs} {inv}"
            if reward > 0:
                positive_chunks.append(action_text)
            elif new_obs or new_inv or obs_changed or inv_changed:
                neutral_chunks.append(action_text)
            if inv_changed or new_inv:
                inventory_chunks.append(f"{action} {inv}")
            if _looks_invalid(obs) or (action and not obs_changed and not inv_changed and reward <= 0):
                failure_chunks.append(action_text)

        proxy["positive_progress_tokens"] = _tokenize(" ".join(positive_chunks[-40:]))
        proxy["neutral_progress_tokens"] = _tokenize(" ".join(neutral_chunks[-60:]))
        proxy["failure_pattern_tokens"] = _tokenize(" ".join(failure_chunks[-60:]))
        proxy["inventory_tokens"] = _tokenize(" ".join(inventory_chunks[-30:]))
        proxy["location_tokens"] = _tokenize(" ".join(observation_heads[-80:]))
        proxy["object_tokens"] = _tokenize(" ".join(object_chunks[-100:]))

    progress_cues = re.findall(
        r"(?:reward|score|inventory|opened|unlocked|found|received|new\s+\w+|"
        r"enter\s+\w+|move\s+\w+|examine\s+\w+|take\s+\w+|drop\s+\w+|"
        r"use\s+\w+|read\s+\w+)\b[^\n]{0,80}",
        text or "",
        flags=re.IGNORECASE,
    )
    proxy["progress_tokens"] = _tokenize(
        " ".join(progress_cues[-50:])
        + " "
        + " ".join(proxy["positive_progress_tokens"])
        + " "
        + " ".join(proxy["neutral_progress_tokens"])
        + " "
        + " ".join(proxy["inventory_tokens"])
        + " "
        + " ".join(proxy["location_tokens"])
        + " "
        + " ".join(proxy["object_tokens"])
    )

    invalid_cues = re.findall(
        r"(?:i don't|cannot|can't|nothing happens|that's not|you can't|"
        r"invalid|not understood|locked|too dark|already)[^\n]{0,80}",
        text or "",
        flags=re.IGNORECASE,
    )
    proxy["invalid_tokens"] = _tokenize(
        " ".join(invalid_cues[-50:]) + " " + " ".join(proxy["failure_pattern_tokens"])
    )

    actions = re.findall(
        r"(?:ACTION TAKEN|ACTION|CHOSEN_ACTION|\[CHOSEN_ACTION\]):\s*([^\n]+)",
        text or "",
        flags=re.IGNORECASE,
    )
    if actions:
        c = Counter(a.strip().lower() for a in actions if a.strip())
        repeated = [a for a, n in c.items() if n >= 3]
        proxy["loop_tokens"] = _tokenize(" ".join(repeated))

    if success_memory:
        try:
            text_pos = " ".join(
                f"{m.get('state', '')} {m.get('action', '')}" for m in success_memory[-10:]
            )
            proxy["success_tokens"] = _tokenize(text_pos)
        except Exception:
            pass
    if failure_memory:
        try:
            parts = []
            for m in failure_memory[-5:]:
                parts.append(" ".join(m.get("actions", []) or []))
                parts.append(" ".join(m.get("states", []) or []))
            proxy["failure_tokens"] = _tokenize(" ".join(parts))
        except Exception:
            pass
    return proxy


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _first_observation_line(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line[:160]
    return ""


def _looks_invalid(text: str) -> bool:
    lowered = (text or "").lower()
    invalid_markers = [
        "i don't",
        "i didnt",
        "i didn't",
        "cannot",
        "can't",
        "you can't",
        "nothing happens",
        "that's not",
        "not a verb",
        "not understood",
        "don't understand",
        "find nothing",
        "no reply",
        "already",
        "can't see",
        "can only do that",
    ]
    return any(marker in lowered for marker in invalid_markers)


def _extract_visible_object_phrases(text: str) -> List[str]:
    phrases = []
    for match in re.finditer(
        r"(?:you can (?:also )?see|there is|there are|contains?|holding|carrying)\s+([^\.\n]{1,120})",
        text or "",
        flags=re.IGNORECASE,
    ):
        phrases.append(match.group(1))
    return phrases


def _parse_transcript_steps(text: str) -> List[Dict[str, str]]:
    if not text:
        return []
    if "Step " in text:
        raw_chunks = re.split(r"\n(?=Step\s+\d+:)", text)
    else:
        raw_chunks = re.split(r"\n=+\n", text)
    chunks = [c for c in raw_chunks if c.strip()]
    steps: List[Dict[str, str]] = []
    for chunk in chunks:
        if "[OBS]" not in chunk and "Step " not in chunk:
            continue
        step = {
            "obs": _extract_bracket_field(chunk, "OBS") or _extract_legacy_field(chunk, "STATE"),
            "inv": _extract_bracket_field(chunk, "INV"),
            "action": (
                _extract_bracket_field(chunk, "CHOSEN_ACTION")
                or _extract_legacy_field(chunk, "ACTION TAKEN")
            ),
            "reward": _extract_bracket_field(chunk, "REWARD")
            or _extract_legacy_field(chunk, "REWARD"),
            "score": _extract_bracket_field(chunk, "CUM_REWARD")
            or _extract_legacy_field(chunk, "SCORE"),
        }
        if step["obs"] or step["action"]:
            steps.append(step)
    return steps


def _extract_bracket_field(chunk: str, label: str) -> str:
    pattern = rf"\[{re.escape(label)}\]\s*(.*?)(?=\n\[[A-Z_]+\]|\n----------|\Z)"
    match = re.search(pattern, chunk, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_legacy_field(chunk: str, label: str) -> str:
    pattern = rf"{re.escape(label)}:\s*(.*?)(?=\n[A-Z][A-Z ]{{2,}}:|\n------------|\Z)"
    match = re.search(pattern, chunk, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def relevance_score(profile_tokens: List[str], proxy: Dict[str, List[str]]) -> float:
    score = 0.0
    score += 1.4 * _jaccard(profile_tokens, proxy.get("progress_tokens", []))
    score += 1.2 * _jaccard(profile_tokens, proxy.get("positive_progress_tokens", []))
    score += 1.0 * _jaccard(profile_tokens, proxy.get("neutral_progress_tokens", []))
    score += 1.2 * _jaccard(profile_tokens, proxy.get("success_tokens", []))
    score += 0.7 * _jaccard(profile_tokens, proxy.get("invalid_tokens", []))
    score += 0.6 * _jaccard(profile_tokens, proxy.get("loop_tokens", []))
    score += 0.8 * _jaccard(profile_tokens, proxy.get("inventory_tokens", []))
    score += 0.8 * _jaccard(profile_tokens, proxy.get("location_tokens", []))
    score += 0.6 * _jaccard(profile_tokens, proxy.get("object_tokens", []))
    if not profile_tokens:
        score *= 0.1
    return _bounded(score, 0.0, 2.0)


def diversity_score(
    profile_tokens: List[str], selected_token_lists: List[List[str]]
) -> float:
    if not selected_token_lists:
        return 1.0
    sims = [_jaccard(profile_tokens, t) for t in selected_token_lists]
    return 1.0 - max(sims) if sims else 1.0


def risk_score(profile: Dict[str, Any], proxy: Dict[str, List[str]]) -> float:
    tokens = _profile_tokens(profile)
    risk = 0.0
    if not tokens:
        risk += 0.5
    if len(tokens) < 5:
        risk += 0.3
    risk += 0.8 * _jaccard(tokens, proxy.get("failure_tokens", []))
    risk += 0.6 * _jaccard(tokens, proxy.get("loop_tokens", []))
    risk += 0.8 * _jaccard(tokens, proxy.get("failure_pattern_tokens", []))
    prompt = profile.get("prompt") or ""
    if isinstance(prompt, str) and len(prompt.strip()) < 20:
        risk += 0.2
    prompt_tokens = Counter(_tokenize(prompt))
    repeated_stall_terms = sum(
        prompt_tokens[t] for t in ("look", "wait", "inventory") if prompt_tokens[t] > 2
    )
    if repeated_stall_terms:
        risk += min(0.25, 0.05 * repeated_stall_terms)
    return _bounded(risk, 0.0, 2.0)


def _short_summary(profile: Dict[str, Any], n: int = 120) -> str:
    p = (profile.get("prompt") or "").strip().replace("\n", " ")
    return p[:n]


def select_candidates(
    parent_config: Any = None,
    candidate_configs: Optional[List[Any]] = None,
    transcript: Any = None,
    score_delta: Optional[float] = None,
    final_score: Optional[float] = None,
    success_memory: Optional[List[Dict[str, Any]]] = None,
    failure_memory: Optional[List[Dict[str, Any]]] = None,
    archived_configs: Optional[List[Any]] = None,
    config_stats: Optional[Dict[Any, Any]] = None,
    top_n: int = 8,
    select_k: int = 4,
    alpha_relevance: float = 1.0,
    beta_diversity: float = 0.5,
    gamma_risk: float = 0.7,
    novelty_weight: float = 0.35,
    debug: bool = False,
    game_name: Optional[str] = None,
    logger=print,
) -> Dict[str, Any]:
    """Re-rank candidate configurations and return a diversified subset.

    Returns a dict with keys ``selected_indices`` (indices into the input
    ``candidate_configs`` list, in DACS-preferred order), ``reordered_configs``
    (the configs themselves in that order), ``scores`` (per-candidate
    breakdown for debugging), ``fallback`` (True iff DACS hit an internal
    error), and ``reason`` (the error message if any).

    On any internal exception the function returns the original candidate
    list unchanged with ``fallback=True`` so callers can keep using the
    original UCB selector.
    """
    try:
        candidate_configs = list(candidate_configs or [])
        if not candidate_configs:
            return {
                "selected_indices": [],
                "reordered_configs": [],
                "scores": [],
                "fallback": False,
                "reason": "",
            }

        pool = list(candidate_configs)
        n_native = len(pool)
        if archived_configs:
            pool.extend(archived_configs)

        proxy = extract_proxy_states(
            transcript,
            score_delta=score_delta,
            final_score=final_score,
            success_memory=success_memory,
            failure_memory=failure_memory,
        )

        profiles = [build_config_profile(c) for c in pool]
        token_lists = [_profile_tokens(p) for p in profiles]

        rel = [relevance_score(tl, proxy) for tl in token_lists]
        risk = [risk_score(p, proxy) for p in profiles]

        scores_prior: List[float] = []
        max_score = 1.0
        for p in profiles:
            s = p.get("score")
            try:
                s_f = float(s) if s is not None else 0.0
            except Exception:
                s_f = 0.0
            if s_f > max_score:
                max_score = s_f
            scores_prior.append(s_f)
        if max_score > 0:
            scores_prior = [s / max_score for s in scores_prior]

        visit_counts: List[int] = []
        for p in profiles:
            raw = p.get("raw")
            try:
                visit_counts.append(len(raw.get("children_idxs", [])) if isinstance(raw, dict) else 0)
            except Exception:
                visit_counts.append(0)

        novelty = [
            novelty_weight
            * (0.5 + 0.5 * rel[i])
            * (1.0 / (1.0 + visit_counts[i]))
            for i in range(len(pool))
        ]

        # initial pruning by combined relevance / risk / prior / novelty
        base = [
            (rel[i] - 0.55 * risk[i] + 0.25 * scores_prior[i] + novelty[i], i)
            for i in range(len(pool))
        ]
        base.sort(key=lambda x: x[0], reverse=True)
        top_idx = [i for _, i in base[: max(top_n, select_k)]]

        # greedy relevance-weighted diversity selection
        selected: List[int] = []
        scored_table: List[Dict[str, Any]] = []
        remaining = list(top_idx)
        while remaining and len(selected) < select_k:
            best_gain = -float("inf")
            best_i = remaining[0]
            best_div = 0.0
            for i in remaining:
                div = diversity_score(
                    token_lists[i], [token_lists[j] for j in selected]
                )
                gain = (
                    alpha_relevance * rel[i]
                    + beta_diversity * div
                    - gamma_risk * risk[i]
                    + 0.2 * scores_prior[i]
                    + novelty[i]
                )
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
                    best_div = div
            selected.append(best_i)
            remaining.remove(best_i)
            scored_table.append(
                {
                    "id": best_i,
                    "relevance": round(rel[best_i], 3),
                    "diversity": round(best_div, 3),
                    "risk": round(risk[best_i], 3),
                    "novelty": round(novelty[best_i], 3),
                    "prior": round(scores_prior[best_i], 3),
                    "visits": visit_counts[best_i],
                    "final_score": round(best_gain, 3),
                    "summary": _short_summary(profiles[best_i]),
                }
            )

        if len(selected) < min(select_k, len(pool)):
            for i in top_idx:
                if i not in selected:
                    selected.append(i)
                if len(selected) >= select_k:
                    break

        # only return indices that point back into the original
        # candidate_configs list (drop archive-only ones for caller safety)
        native_selected = [i for i in selected if i < n_native]
        if not native_selected:
            native_selected = list(range(min(select_k, n_native)))
        reordered = [candidate_configs[i] for i in native_selected]

        all_scores = []
        selected_set = set(native_selected)
        for i in range(n_native):
            div = diversity_score(token_lists[i], [token_lists[j] for j in native_selected if j != i])
            final = (
                alpha_relevance * rel[i]
                + beta_diversity * div
                - gamma_risk * risk[i]
                + 0.2 * scores_prior[i]
                + novelty[i]
            )
            all_scores.append(
                {
                    "id": i,
                    "relevance": round(rel[i], 3),
                    "diversity": round(div, 3),
                    "risk": round(risk[i], 3),
                    "novelty": round(novelty[i], 3),
                    "prior": round(scores_prior[i], 3),
                    "visits": visit_counts[i],
                    "final_score": round(final, 3),
                    "selected": i in selected_set,
                    "summary": _short_summary(profiles[i]),
                }
            )

        if debug:
            try:
                logger(f"[DACS] enabled=True game={game_name}")
                logger(
                    f"[DACS] num raw candidates = {n_native} "
                    f"archived={len(archived_configs or [])}"
                )
                logger(f"[DACS] num filtered candidates = {len(native_selected)}")
                logger(f"[DACS] selected/reordered candidate ids = {native_selected}")
                proxy_counts = {
                    key: len(value)
                    for key, value in proxy.items()
                    if key.endswith("_tokens") or key in ("progress_tokens", "invalid_tokens")
                }
                logger(f"[DACS] proxy token counts = {proxy_counts}")
                logger("[DACS] candidate score table:")
                logger(
                    "  id | relevance | diversity | risk | novelty | prior | visits | "
                    "final_score | selected | summary"
                )
                for row in all_scores:
                    logger(
                        f"  {row['id']} | {row['relevance']} | {row['diversity']} | "
                        f"{row['risk']} | {row['novelty']} | {row['prior']} | "
                        f"{row['visits']} | {row['final_score']} | {row['selected']} | "
                        f"{row['summary'][:80]}"
                    )
                logger("[DACS] fallback=False")
            except Exception:
                pass

        return {
            "selected_indices": native_selected,
            "reordered_configs": reordered,
            "scores": all_scores,
            "selected_scores": scored_table,
            "fallback": False,
            "reason": "",
        }
    except Exception as e:
        if debug:
            try:
                logger(f"[DACS] fallback=True reason={e}")
                logger(traceback.format_exc())
            except Exception:
                pass
        return {
            "selected_indices": list(range(len(candidate_configs or []))),
            "reordered_configs": list(candidate_configs or []),
            "scores": [],
            "fallback": True,
            "reason": str(e),
        }
