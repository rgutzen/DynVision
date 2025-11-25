"""Shared utilities for loading and resolving configuration modes."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import yaml

from dynvision.project_paths import project_paths
from dynvision.utils import str_to_bool

logger = logging.getLogger(__name__)

ModeDetector = Callable[[Mapping[str, Any]], bool]


@dataclass(frozen=True)
class ModeDefinition:
    """Static metadata describing a mode entry from config_modes.yaml."""

    name: str
    toggle_key: str
    default_toggle: Any
    payload: Dict[str, Any]

    def payload_copy(self) -> Dict[str, Any]:
        """Return a deep copy of the mode payload to avoid accidental mutation."""

        return copy.deepcopy(self.payload)


@dataclass(frozen=True)
class ModeResolution:
    """Resolved activation state for all modes."""

    toggles: Dict[str, bool]
    raw_values: Dict[str, Any]
    active_modes: Tuple[str, ...]
    patches: Dict[str, Dict[str, Any]]


class ModeRegistry:
    """Utility providing access to mode definitions and activation logic."""

    _CONFIG_FILENAME = "config_modes.yaml"

    # Order matters for reproducible metadata headers
    _DEFAULT_DETECTORS: Dict[str, ModeDetector] = {}

    @classmethod
    def config_path(cls) -> Path:
        """Return the fully-qualified config_modes.yaml path."""

        return project_paths.scripts.configs / cls._CONFIG_FILENAME

    @classmethod
    @lru_cache(maxsize=1)
    def _load_raw_config(cls) -> Dict[str, Any]:
        """Load the raw YAML document once and cache it."""

        cfg_path = cls.config_path()
        if not cfg_path.exists():
            logger.warning("Mode config file not found at %s", cfg_path)
            return {}

        with cfg_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        if not isinstance(data, dict):
            logger.warning(
                "Unexpected structure in config_modes.yaml; expected mapping"
            )
            return {}
        return data

    @classmethod
    @lru_cache(maxsize=1)
    def _definitions(cls) -> Dict[str, ModeDefinition]:
        """Parse all mode definitions from the raw config file."""

        raw = cls._load_raw_config()
        definitions: Dict[str, ModeDefinition] = {}
        for key, value in raw.items():
            if not key.startswith("use_") or not key.endswith("_mode"):
                continue

            # Keep the `_mode` suffix in the public mode name so payload keys
            # and human-readable sections align one-to-one (e.g., local_mode).
            mode_name = key[len("use_") :]

            payload = raw.get(mode_name, {})
            if payload is None:
                payload = {}
            if not isinstance(payload, dict):
                logger.warning(
                    "Mode payload for %s is not a mapping; ignoring", mode_name
                )
                payload = {}
            definitions[mode_name] = ModeDefinition(
                name=mode_name,
                toggle_key=key,
                default_toggle=value,
                payload=payload,
            )
        return definitions

    @classmethod
    def list_modes(cls) -> Tuple[str, ...]:
        """Return the ordered tuple of known mode names."""

        return tuple(cls._definitions().keys())

    @classmethod
    def get_definition(cls, mode_name: str) -> Optional[ModeDefinition]:
        """Return the mode definition if it exists."""

        return cls._definitions().get(mode_name)

    @classmethod
    def reload(cls) -> None:
        """Clear cached data so future calls re-read config_modes.yaml."""

        cls._load_raw_config.cache_clear()
        cls._definitions.cache_clear()

    @classmethod
    def register_detector(cls, mode_name: str, detector: ModeDetector) -> None:
        """Register or override the detector used when a mode toggle is 'auto'."""

        cls._DEFAULT_DETECTORS[mode_name] = detector

    @classmethod
    def _get_detector(cls, mode_name: str) -> Optional[ModeDetector]:
        """Return the detector callable for the mode, if any."""

        if not cls._DEFAULT_DETECTORS:
            cls._initialize_default_detectors()
        return cls._DEFAULT_DETECTORS.get(mode_name)

    @classmethod
    def resolve_modes(
        cls,
        toggle_source: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> ModeResolution:
        """Resolve activation state and patches for all configured modes."""

        definitions = cls._definitions()
        toggles: Dict[str, bool] = {}
        raw_values: Dict[str, Any] = {}
        patches: Dict[str, Dict[str, Any]] = {}
        active_modes: Tuple[str, ...] = tuple()

        active: list[str] = []
        for mode_name, definition in definitions.items():
            raw_value = toggle_source.get(
                definition.toggle_key, definition.default_toggle
            )
            raw_values[mode_name] = raw_value
            active_flag = cls._evaluate_toggle(mode_name, raw_value, context)
            toggles[mode_name] = active_flag
            if active_flag:
                active.append(mode_name)
                patches[mode_name] = definition.payload_copy()

        active_modes = tuple(active)
        return ModeResolution(
            toggles=toggles,
            raw_values=raw_values,
            active_modes=active_modes,
            patches=patches,
        )

    @classmethod
    def _evaluate_toggle(
        cls,
        mode_name: str,
        raw_value: Any,
        context: Mapping[str, Any],
    ) -> bool:
        """Evaluate a toggle value, honoring 'auto' semantics and detectors."""

        normalized = cls._normalize_toggle(raw_value)
        if isinstance(normalized, str) and normalized.lower() == "auto":
            detector = cls._get_detector(mode_name)
            if detector is None:
                logger.debug(
                    "Mode '%s' requested auto but no detector configured", mode_name
                )
                return False
            try:
                return bool(detector(context))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Mode detector '%s' failed: %s", mode_name, exc)
                return False
        return bool(normalized)

    @staticmethod
    def _normalize_toggle(value: Any) -> Any:
        """Normalize toggle values by coercing strings to python types."""

        if isinstance(value, str):
            lower = value.strip().lower()
            if lower == "auto":
                return "auto"
            return str_to_bool(value)
        return value

    @classmethod
    def _initialize_default_detectors(cls) -> None:
        """Populate the default detector map. Separated for testability."""

        cls._DEFAULT_DETECTORS = {
            "local_mode": _detect_local_mode,
            "debug_mode": _detect_debug_mode,
            "large_dataset_mode": _detect_large_dataset_mode,
            "distributed_mode": _detect_distributed_mode,
        }


def _ensure_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, str):
        try:
            return str_to_bool(value)
        except ValueError:
            return default
    return bool(value) if value is not None else default


def _detect_local_mode(context: Mapping[str, Any]) -> bool:
    local_execution = context.get("local_execution")
    if local_execution is None:
        # Fall back to host detection when explicit metadata is absent
        return not project_paths.iam_on_cluster()
    return _ensure_bool(local_execution, default=True)


def _detect_debug_mode(context: Mapping[str, Any]) -> bool:
    log_level = str(context.get("log_level", "")).upper()
    if log_level == "DEBUG":
        return True
    epochs = context.get("epochs")
    if epochs is None:
        return False
    try:
        return int(epochs) <= 5
    except (TypeError, ValueError):
        return False


def _detect_large_dataset_mode(context: Mapping[str, Any]) -> bool:
    data_name = str(context.get("data_name", "")).lower()
    if not data_name:
        return False
    large_datasets = ("imagenet", "coco", "openimages")
    return any(token in data_name for token in large_datasets)


def _detect_distributed_mode(context: Mapping[str, Any]) -> bool:
    local_execution = _ensure_bool(context.get("local_execution"), default=True)
    if local_execution:
        return False
    return _ensure_bool(context.get("use_distributed"), default=False)
