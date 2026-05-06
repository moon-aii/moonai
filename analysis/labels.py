"""Condition grouping for MoonAI analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .io import RunData


SEED_SUFFIX_RE = re.compile(r"^(?P<label>.+)_seed(?P<seed>\d+)$")
RETRY_SUFFIX_RE = re.compile(r"^(?P<label>.+)_(?P<retry>\d+)$")


@dataclass
class LabelResolver:
    signature_labels: dict[str, str]
    assigned_counts: dict[str, int]

    def __init__(self) -> None:
        self.signature_labels = {}
        self.assigned_counts = {}

    def resolve(self, run: RunData) -> str:
        by_name = self._from_name(run.name)
        if by_name is not None:
            self.signature_labels.setdefault(run.config_signature, by_name)
            return by_name

        by_signature = self.signature_labels.get(run.config_signature)
        if by_signature is not None:
            return by_signature

        label = self._fallback_label(run.name)
        count = self.assigned_counts.get(label, 0)
        self.assigned_counts[label] = count + 1
        unique = label if count == 0 else f"{label}_{count + 1}"
        self.signature_labels[run.config_signature] = unique
        return unique

    @staticmethod
    def _from_name(name: str) -> str | None:
        seed_match = SEED_SUFFIX_RE.match(name)
        if seed_match:
            return seed_match.group("label")

        retry_match = RETRY_SUFFIX_RE.match(name)
        if retry_match:
            return retry_match.group("label")

        if name:
            return name
        return None

    @staticmethod
    def _fallback_label(name: str) -> str:
        return name or "condition"
