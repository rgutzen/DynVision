"""Utilities for consistent logging across DynVision."""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

DEFAULT_FORMAT = "%(asctime)s | %(levelname)8s | %(name)s | %(message)s"
_LogValue = Union[str, int, float, bool]


def configure_logging(
    level: Union[str, int] = "INFO",
    *,
    logger_name: Optional[str] = None,
    log_file: Optional[Path] = None,
    force: bool = True,
) -> logging.Logger:
    """Configure root logging once using a consistent format.

    Args:
        level: Logging level name or integer value.
        logger_name: Optional logger name to return after configuration.
        log_file: Optional file to tee logs into in addition to console.
        force: Whether to override existing configuration (defaults to True).

    Returns:
        Configured logger instance (root when logger_name is None).
    """

    log_level = (
        level
        if isinstance(level, int)
        else (
            logging.getLevelName(level.upper())
            if isinstance(level, str)
            else logging.INFO
        )
    )

    # Basic configuration for the root logger
    logging.basicConfig(level=log_level, format=DEFAULT_FORMAT, force=force)

    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    logger.setLevel(log_level)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if not any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == log_path
            for handler in logger.handlers
        ):
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
            logger.addHandler(file_handler)

    return logger


def format_value(value: Any, *, max_length: int = 120, max_items: int = 6) -> str:
    """Format an arbitrary value for log output in a compact representation."""

    if isinstance(value, Path):
        text = str(value)
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
        preview = ", ".join(
            format_value(item, max_length=max_length // 2)
            for item in items[:max_items]
        )
        if len(items) > max_items:
            preview += ", …"
        text = f"[{preview}]" if isinstance(value, list) else f"({preview})"
    elif isinstance(value, dict):
        items = list(value.items())
        preview = ", ".join(
            f"{k}: {format_value(v, max_length=max_length // 2)}"
            for k, v in items[:max_items]
        )
        if len(items) > max_items:
            preview += ", …"
        text = f"{{{preview}}}"
    elif value is None:
        text = "–"
    else:
        text = str(value)

    if len(text) > max_length:
        return text[: max_length - 1] + "…"
    return text


def log_section(
    logger: logging.Logger,
    title: str,
    entries: Sequence[Tuple[str, str, Optional[str]]],
    *,
    level: int = logging.INFO,
    indent: str = "  ",
) -> None:
    """Log a titled section with aligned key-value pairs."""

    if not entries or not logger.isEnabledFor(level):
        return

    logger.log(level, "%s:", title)
    for label, value, marker in entries:
        suffix = f" ({marker})" if marker else ""
        logger.log(level, "%s- %s: %s%s", indent, label, value, suffix)


SummarySource = Union[str, Callable[[Any], Any]]


@dataclass(frozen=True)
class SummaryItem:
    """Specification of a single entry in a parameter logging summary."""

    source: SummarySource
    label: Optional[str] = None
    formatter: Optional[Callable[[Any], str]] = None
    always: bool = False

    def resolve(self, instance: Any) -> Tuple[Optional[str], Any]:
        """Resolve label and value for the given instance."""

        label = self.label
        if label is None:
            if isinstance(self.source, str):
                label = self.source
            else:
                label = "value"

        if isinstance(self.source, str):
            value = getattr(instance, self.source, None)
        else:
            value = self.source(instance)

        return label, value

    def format(self, value: Any) -> str:
        if self.formatter:
            try:
                return self.formatter(value)
            except Exception:  # pragma: no cover - defensive formatting fallback
                return str(value)
        return format_value(value)

    @property
    def key(self) -> Optional[str]:
        return self.source if isinstance(self.source, str) else None


def build_section(
    instance: Any,
    items: Iterable[SummaryItem],
    *,
    include_defaults: bool,
    override_keys: Iterable[str],
    dynamic_updates: Iterable[str],
) -> List[Tuple[str, str, Optional[str]]]:
    """Render a sequence of summary items to formatted log entries."""

    override_set = set(override_keys)
    dynamic_set = set(dynamic_updates)
    provenance_getter = getattr(instance, "get_provenance", None)
    entries: List[Tuple[str, str, Optional[str]]] = []

    for item in items:
        label, value = item.resolve(instance)
        if value is None and not item.always:
            continue
        if value in ("", [], {}, ()):  # Empty containers
            if not item.always:
                continue

        formatted = item.format(value)
        marker: Optional[str] = None
        key = item.key
        provenance_record = None
        if key and callable(provenance_getter):
            try:
                provenance_record = provenance_getter(key)
            except Exception:  # pragma: no cover - defensive fallback
                provenance_record = None

        if provenance_record is not None:
            if (
                not include_defaults
                and provenance_record.is_default()
                and not item.always
            ):
                continue
            marker = provenance_record.format()

        if marker is None and key:
            if key in override_set:
                marker = "override"
            elif key in dynamic_set:
                marker = "runtime"

        if not include_defaults and marker is None and not item.always:
            continue

        entries.append((label, formatted, marker))

    return entries


def resolve_signature_defaults(
    callable_obj: Any,
    provided_kwargs: Dict[str, Any],
) -> Tuple["OrderedDict[str, Any]", Dict[str, bool]]:
    """Resolve kwargs against a callable's signature, filling in defaults.

    Args:
        callable_obj: Function, method, or class whose signature should be inspected.
        provided_kwargs: Keyword arguments explicitly supplied by the caller.

    Returns:
        A tuple containing an ordered mapping of parameter names to effective values
        (explicit or default) and a dict mapping parameter names to a boolean flag
        indicating whether the value came from the callable's default (True) or was
        explicitly provided (False).
    """

    if inspect.isclass(callable_obj):
        target = callable_obj.__init__
    else:
        target = callable_obj

    signature = inspect.signature(target)
    resolved: "OrderedDict[str, Any]" = OrderedDict()
    default_flags: Dict[str, bool] = {}

    for name, parameter in signature.parameters.items():
        if name == "self" or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        if name in provided_kwargs:
            resolved[name] = provided_kwargs[name]
            default_flags[name] = False
        elif parameter.default is not inspect._empty:
            resolved[name] = parameter.default
            default_flags[name] = True

    for name, value in provided_kwargs.items():
        if name not in resolved:
            resolved[name] = value
            default_flags[name] = False

    return resolved, default_flags
