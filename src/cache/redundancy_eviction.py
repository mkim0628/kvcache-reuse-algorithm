"""RedundancyAwareEvictionPolicy — dual-score (importance × redundancy) eviction (Activity C aux).

Scores TTL-expired segments by:
    eviction_score = (1 - normalized_importance) × redundancy_score

Higher score → evicted first. High-importance segments (importance ≈ 1.0) get
eviction_score ≈ 0 and are never selected. Redundant segments (cosine sim ≈ 1.0
to other cached segments) are evicted before unique ones.

This class is a pure scoring layer; it does NOT inherit from CacheStore.
"""

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from src.cache.workload_ttl_cache import TTLEntry


class RedundancyAwareEvictionPolicy:
    """Dual-score eviction policy used as a hook in WorkloadAwareTTLCache.

    Attributes:
        redundancy_top_n: Maximum number of candidates for brute-force similarity.
        importance_weight: Multiplier on importance term (kept at 1.0 normally).
        redundancy_weight: Multiplier on redundancy term (kept at 1.0 normally).
        doc_id_shortcut: If True, segments sharing a 'doc:<id>:' key prefix
                         get redundancy=1.0 immediately (O(1) shortcut).
    """

    def __init__(
        self,
        redundancy_top_n: int = 100,
        importance_weight: float = 1.0,
        redundancy_weight: float = 1.0,
        doc_id_shortcut: bool = True,
    ) -> None:
        self.redundancy_top_n = redundancy_top_n
        self.importance_weight = importance_weight
        self.redundancy_weight = redundancy_weight
        self.doc_id_shortcut = doc_id_shortcut

    def score_candidates(
        self,
        candidates: List[str],
        store_entries: Dict[str, "TTLEntry"],
    ) -> List[Tuple[str, float]]:
        """Score eviction candidates; returns (key, eviction_score) sorted descending.

        eviction_score = (1 - normalized_importance) × redundancy_score
        """
        if not candidates:
            return []

        # --- Step 1: importance normalisation ---
        importances = {
            k: store_entries[k].importance_score
            for k in candidates
            if k in store_entries
        }
        max_imp = max(importances.values()) if importances else 1.0
        if max_imp == 0.0:
            max_imp = 1.0
        norm_imp = {k: v / max_imp for k, v in importances.items()}

        # --- Step 2: redundancy via doc_id shortcut + cosine similarity ---
        redundancy: Dict[str, float] = {k: 0.0 for k in candidates}

        if self.doc_id_shortcut:
            self._apply_doc_id_shortcut(candidates, redundancy)

        # Collect candidates that still need embedding-based redundancy
        need_emb = [
            k for k in candidates
            if k in store_entries and store_entries[k].embedding is not None
            and redundancy[k] < 1.0  # skip already-shortcut ones
        ]
        need_emb = need_emb[: self.redundancy_top_n]

        if len(need_emb) >= 2:
            embeddings = torch.stack([store_entries[k].embedding for k in need_emb])  # (N, d)
            e_norm = F.normalize(embeddings, dim=-1)
            sim_matrix = e_norm @ e_norm.T  # (N, N)
            sim_matrix.fill_diagonal_(0.0)
            # Mean similarity to all other candidates (excluding self)
            mean_sim = sim_matrix.mean(dim=-1)  # (N,)
            for i, k in enumerate(need_emb):
                redundancy[k] = float(mean_sim[i].clamp(min=0.0).item())

        # --- Step 3: eviction score ---
        scored: List[Tuple[str, float]] = []
        for k in candidates:
            imp = norm_imp.get(k, 0.0) * self.importance_weight
            red = redundancy[k] * self.redundancy_weight
            score = (1.0 - imp) * red
            scored.append((k, score))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    def select_evict_keys(
        self,
        candidates: List[str],
        store_entries: Dict[str, "TTLEntry"],
        n_evict: int = 1,
    ) -> List[str]:
        """Return the top n_evict keys by eviction score."""
        scored = self.score_candidates(candidates, store_entries)
        return [k for k, _ in scored[:n_evict]]

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _apply_doc_id_shortcut(
        self,
        candidates: List[str],
        redundancy: Dict[str, float],
    ) -> None:
        """Set redundancy=1.0 for any two segments sharing the same doc_id prefix.

        Expected key format: 'doc:<doc_id>:<rest>' or any key starting with
        the same colon-delimited prefix.
        """
        prefix_groups: Dict[str, List[str]] = {}
        for k in candidates:
            parts = k.split(":")
            if len(parts) >= 2 and parts[0] == "doc":
                doc_prefix = f"doc:{parts[1]}"
                prefix_groups.setdefault(doc_prefix, []).append(k)

        for group in prefix_groups.values():
            if len(group) >= 2:
                for k in group:
                    redundancy[k] = 1.0
