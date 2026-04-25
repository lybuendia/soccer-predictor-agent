"""Canonical team-name normalization backed by a YAML alias map."""

from pathlib import Path
import re

import yaml


class TeamNameNormalizer:
    """Map external team-name variations to one canonical internal name."""

    def __init__(self, aliases_path: Path | None = None) -> None:
        """Load canonical names and aliases from YAML."""
        self._aliases_path = aliases_path or Path(__file__).with_name("team_aliases.yaml")
        self._canonical_by_alias = self._load_alias_map()

    def canonicalize(self, name: str) -> str:
        """Return the canonical name for an alias, or the stripped input if unknown."""
        if not isinstance(name, str):
            raise ValueError("Team name must be a string.")
        stripped = name.strip()
        if not stripped:
            raise ValueError("Team name cannot be empty.")
        return self._canonical_by_alias.get(self._normalize_key(stripped), stripped)

    def _load_alias_map(self) -> dict[str, str]:
        """Return alias -> canonical mappings built from the YAML file."""
        data = yaml.safe_load(self._aliases_path.read_text()) or {}
        mappings: dict[str, str] = {}
        for canonical, aliases in data.items():
            all_names = [canonical, *(aliases or [])]
            for alias in all_names:
                mappings[self._normalize_key(alias)] = canonical
        return mappings

    def _normalize_key(self, name: str) -> str:
        """Return a forgiving normalization key for team-name lookups."""
        lowered = name.casefold().replace("&", " and ")
        lowered = re.sub(r"\bfc\b", "", lowered)
        lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()
