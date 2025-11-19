"""Shared utilities for composite parameter classes."""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
)
import logging
from datetime import datetime
from pathlib import Path
from enum import Enum

import yaml

try:  # pragma: no cover - optional dependency guard
    import torch
except Exception:  # pragma: no cover - torch unavailable in some doc builds
    torch = None

from dynvision.params.base_params import (
    BaseParams,
    DynVisionValidationError,
    ParamsDict,
    ProvenanceRecord,
)
from dynvision.params.mode_registry import ModeRegistry, ModeResolution

logger = logging.getLogger(__name__)


ComponentDict = Dict[str, Any]
Preprocessor = Callable[[ComponentDict], ComponentDict]


class CompositeParams(BaseParams):
    """
    Base class for parameter objects composed of multiple component parameter sets.

    Subclasses must define :pyattr:`component_classes` mapping field names to the
    underlying ``BaseParams`` subclasses. Optional preprocessors can mutate the raw
    dictionaries before component instantiation. Unknown parameters can be handled
    by overriding :py:meth:`_handle_unscoped_param` or
    :py:meth:`_assign_to_components`.

    Mode-Specific Parameter Precedence:

    Subclasses can define :pyattr:`mode_name` to enable mode-specific parameter
    precedence. Parameters are resolved with the following hierarchy (highest to lowest):

    1. CLI arguments (handled upstream)
    2. mode.component.param (e.g., test.data.batch_size)
    3. mode.param (e.g., test.batch_size)
    4. component.param (e.g., data.batch_size)
    5. param (base defaults)

    Example:
        class TestingParams(CompositeParams):
            mode_name = "test"
            component_classes = {"data": DataParams, "trainer": TrainerParams}
    """

    # Mapping of component field name -> BaseParams subclass
    component_classes: ClassVar[Dict[str, Type[BaseParams]]] = {}

    # Optional mode name for mode-specific parameter precedence
    # ClassVar indicates this is a class variable, not a Pydantic instance field
    mode_name: ClassVar[Optional[str]] = None

    @classmethod
    def get_component_classes(cls) -> Dict[str, Type[BaseParams]]:
        """Return component mapping for the composite configuration."""
        if not cls.component_classes:
            raise NotImplementedError(
                f"{cls.__name__} must define component_classes to use CompositeParams"
            )
        return cls.component_classes

    @classmethod
    def get_component_preprocessors(cls) -> Dict[str, Preprocessor]:
        """
        Optional per-component preprocessors applied before instantiation.

        Returns a mapping of component name to callable that receives the component
        parameter dictionary and returns the possibly modified dictionary.
        """
        return {}

    @classmethod
    def get_component_assignment_order(cls) -> Iterable[str]:
        """
        Ordering used when attributing unscoped keys to components.

        Default ordering is the declaration order of ``component_classes``.
        """
        return cls.get_component_classes().keys()

    def iter_components(self) -> Iterable[Tuple[str, BaseParams]]:
        """Iterate over instantiated component parameter objects."""

        for name in self.get_component_classes().keys():
            component = getattr(self, name, None)
            if isinstance(component, BaseParams):
                yield name, component

    def log_overview(
        self,
        *,
        logger: Optional[logging.Logger] = None,
        include_components: bool = True,
        include_defaults: bool = False,
    ) -> None:
        """Log a summarized view of the composite and its components."""

        logger = logger or logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

        legend_requested = bool(
            self.show_provenance_legend or getattr(self, "verbose", False)
        )

        self.log_summary(
            logger=logger,
            title=f"{self.__class__.__name__} parameters",
            include_defaults=include_defaults,
            force_provenance_legend=legend_requested,
        )

        if include_components:
            for name, component in self.iter_components():
                component.log_summary(
                    logger=logger.getChild(name),
                    title=f"{name.capitalize()} parameters",
                    include_defaults=include_defaults,
                    force_provenance_legend=legend_requested,
                )

    @classmethod
    def _get_active_mode(cls) -> Optional[str]:
        """Return the active mode name if defined."""
        return cls.mode_name

    @classmethod
    def _load_config_file(cls, config_path) -> Dict[str, Any]:
        """
        Load config file and apply mode-specific overrides.

        Merges mode-specific sections (e.g., test.data.train) into base config.
        Mode name comes from cls.mode_name (e.g., "test" for TestingParams).

        Example config structure:
            train: true          # base default
            test:
              data:
                train: false    # test mode override

        For TestingParams (mode_name="test"), this merges test.data.train into data.train.
        """
        # Load base config using parent method
        config = super()._load_config_file(config_path)

        # Apply mode-specific overrides if mode is defined
        mode = cls._get_active_mode()
        if mode and mode in config:
            mode_entry = config.get(mode)

            # Only treat the mode key as overrides when it is a dictionary. Primitive
            # values (e.g., the legacy root-level `train: true`) must remain in the
            # config so they continue to act as component defaults.
            if isinstance(mode_entry, dict):
                mode_overrides = config.pop(mode)

                # Deep merge mode overrides into base config
                cls._deep_merge(config, mode_overrides)

                # Resolve conflicts between base-level and scoped keys
                # E.g., if test.data.train was merged into data.train, remove base-level train
                cls._remove_conflicting_base_keys(config, mode_overrides)

                # Flatten nested component sections back to dotted notation
                # _separate_single_source expects flat keys like "trainer.devices", not nested dicts
                config = cls._flatten_component_sections(config)

        return config

    @classmethod
    def _deep_merge(cls, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """
        Deep merge override dict into base dict (modifies base in-place).

        Args:
            base: Base dictionary to merge into
            override: Override dictionary to merge from
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                cls._deep_merge(base[key], value)
            else:
                # Override value
                base[key] = value

    @classmethod
    def _flatten_component_sections(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten nested component sections back to dotted notation.

        _separate_single_source expects flat keys like "trainer.devices",
        but after deep_merge and conflict resolution we have nested dicts
        like config['trainer']['devices'].

        This method converts:
            {'trainer': {'devices': 1, 'num_nodes': 1}, 'seed': 42}
        To:
            {'trainer.devices': 1, 'trainer.num_nodes': 1, 'seed': 42}

        Only flattens known component sections, leaves other nested dicts intact.

        Args:
            config: Config dict with potentially nested component sections

        Returns:
            Flattened config dict with dotted keys
        """
        component_names = set(cls.get_component_classes().keys())
        flattened = {}

        for key, value in config.items():
            if key in component_names and isinstance(value, dict):
                # This is a component section - flatten it
                for subkey, subvalue in value.items():
                    flattened[f"{key}.{subkey}"] = subvalue
            else:
                # Not a component section, keep as-is
                flattened[key] = value

        return flattened

    @classmethod
    def _remove_conflicting_base_keys(
        cls, config: Dict[str, Any], mode_overrides: Dict[str, Any]
    ) -> None:
        """
        Resolve conflicts between base-level keys and scoped mode overrides.

        Instead of removing base-level keys, this method applies precedence rules:
        1. Mode-specific scoped value (e.g., test.trainer.devices) - highest
        2. Existing scoped value (e.g., trainer.devices) - medium
        3. Base-level value (e.g., devices) - lowest

        If a base-level key conflicts with a mode override, we ensure the scoped
        section has the correct value and can safely use the base-level as fallback.

        Args:
            config: Base config dictionary to resolve (modified in-place)
            mode_overrides: Mode-specific overrides that were merged
        """

        def resolve_conflicts(overrides: Dict[str, Any], scope: str = "") -> None:
            """
            Recursively resolve conflicts by ensuring scoped sections have values.

            For each mode override like 'trainer.devices = 1':
            1. Ensure config['trainer']['devices'] = 1 (from mode merge)
            2. If base-level 'devices' exists, ensure scoped version takes precedence
            3. If scoped section missing the key, copy from base-level (fallback)
            """
            for key, value in overrides.items():
                if isinstance(value, dict):
                    # This is a scope (e.g., 'data', 'trainer')
                    # Ensure the scope exists in config
                    if key not in config:
                        config[key] = {}
                    elif not isinstance(config[key], dict):
                        # Base-level value exists but scope needs to be a dict
                        # Convert it to a dict (this is rare/unusual)
                        config[key] = {}

                    # Recurse into the scope
                    resolve_conflicts(value, key)
                else:
                    # This is a leaf key within a scope
                    if scope:
                        # Ensure the scoped section exists
                        if scope not in config:
                            config[scope] = {}
                        elif not isinstance(config[scope], dict):
                            config[scope] = {}

                        # Apply precedence: mode override > scoped value > base-level value
                        if key in config and not isinstance(config[key], dict):
                            # Base-level key exists - use it as fallback if scoped version doesn't exist
                            base_value = config[key]

                            if key not in config[scope]:
                                # Copy base-level value to scoped section as fallback
                                config[scope][key] = base_value

                            # Remove base-level key to prevent conflict during component separation
                            config.pop(key)
                        else:
                            # No base-level conflict, mode override already applied
                            if key not in config[scope]:
                                # Shouldn't happen after deep_merge, but ensure scoped key exists
                                config[scope][key] = value

        # Resolve all conflicts
        resolve_conflicts(mode_overrides)

    @classmethod
    def _resolve_aliases_with_precedence(
        cls,
        params: Dict[str, Any],
        sources: Optional[Dict[str, ProvenanceRecord]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, ProvenanceRecord]]:
        """
        Resolve aliases with scope-aware precedence.

        Precedence rules:
        1. Scope level is primary: scoped > unscoped (target wins if scoped)
        2. Within same scope: alias > long form
        3. Cross-scope: only resolve if target doesn't exist

        Examples:
            - model.tff + model.t_feedforward → model.t_feedforward = tff_value (same scope, alias wins)
            - tff + t_feedforward → t_feedforward = tff_value (same scope, alias wins)
            - model.t_feedforward + tff → model.t_feedforward keeps its value (target scoped, beats unscoped alias)
            - tsteps → model.n_timesteps (cross-scope, resolve if target absent)

        Args:
            params: Parameters with potential aliases

        Returns:
            Parameters with aliases resolved respecting scope precedence
        """
        aliases = cls.get_aliases()
        if not aliases:
            return params, (sources or {})

        resolved = params.copy()
        resolved_sources: Dict[str, ProvenanceRecord] = dict(sources or {})

        # Process aliases grouped by relationship:
        # 1. Same-scope aliases (alias and target at same scope level)
        # 2. Cross-scope aliases (alias and target at different scope levels)

        same_scope_aliases = []
        cross_scope_aliases = []

        for alias, target in aliases.items():
            alias_scope = alias.count(".")
            target_scope = target.count(".")
            if alias_scope == target_scope:
                same_scope_aliases.append((alias, target, alias_scope))
            else:
                cross_scope_aliases.append((alias, target, alias_scope, target_scope))

        # === Phase 1: Resolve same-scope aliases (alias wins over target) ===
        for alias, target, scope_level in same_scope_aliases:
            if alias in resolved:
                alias_value = resolved[alias]
                if target in resolved:
                    # Both exist at same scope - alias wins
                    logger.debug(
                        f"Alias '{alias}'={alias_value} overrides '{target}'={resolved[target]} "
                        f"(same scope level {scope_level}, alias wins)"
                    )
                else:
                    # Only alias exists - resolve it
                    logger.debug(
                        f"Alias '{alias}'={alias_value} resolves to '{target}' (same scope level {scope_level})"
                    )
                resolved[target] = alias_value
                if alias in resolved_sources:
                    resolved_sources[target] = resolved_sources[alias]
                resolved_sources.pop(alias, None)
                del resolved[alias]

        # === Phase 2: Resolve cross-scope aliases (scoped target beats unscoped alias) ===
        for alias, target, alias_scope, target_scope in cross_scope_aliases:
            if alias in resolved:
                alias_value = resolved[alias]
                if target in resolved:
                    # Target exists - which has precedence?
                    if target_scope > alias_scope:
                        # Target is more scoped (higher precedence) - keep target
                        logger.debug(
                            f"Scoped target '{target}'={resolved[target]} beats unscoped alias '{alias}'={alias_value} "
                            f"(target scope {target_scope} > alias scope {alias_scope})"
                        )
                    else:
                        # Alias is more scoped (higher precedence) - use alias
                        logger.debug(
                            f"Scoped alias '{alias}'={alias_value} overrides unscoped target '{target}'={resolved[target]} "
                            f"(alias scope {alias_scope} > target scope {target_scope})"
                        )
                        resolved[target] = alias_value
                        if alias in resolved_sources:
                            resolved_sources[target] = resolved_sources[alias]
                    # Remove alias in both cases
                    resolved_sources.pop(alias, None)
                    del resolved[alias]
                else:
                    # Only alias exists - resolve it to target
                    logger.debug(
                        f"Cross-scope alias '{alias}'={alias_value} resolves to '{target}' "
                        f"(alias scope {alias_scope}, target scope {target_scope})"
                    )
                    resolved[target] = alias_value
                    if alias in resolved_sources:
                        resolved_sources[target] = resolved_sources[alias]
                    resolved_sources.pop(alias, None)
                    del resolved[alias]
        return resolved, resolved_sources

    @classmethod
    def from_cli_and_config(
        cls,
        config_path: Optional[str] = None,
        override_kwargs: Optional[Dict[str, Any]] = None,
        args: Optional[List[str]] = None,
    ) -> "CompositeParams":
        """Create an instance with component-aware parameter separation.

        Applies proper precedence:
        - Within config files: scoped > unscoped (model.model_name > model_name)
        - Within CLI args: scoped > unscoped (model.model_name > model_name)
        - Between sources: ALL CLI args > ALL config values
        """
        # Get config params and CLI params separately
        config_params, cli_params = cls._get_config_and_cli_params_separate(
            config_path=config_path, override_kwargs=override_kwargs, args=args
        )

        # Resolve mode overrides after both sources are parsed
        mode_params, mode_resolution = cls._resolve_mode_overrides(
            config_params, cli_params
        )

        # Remove toggle keys so they do not leak into component configs
        cls._strip_mode_toggle_keys(config_params, cli_params)

        # Separate each source independently, then merge with CLI taking precedence
        separated = cls._separate_component_configs_two_sources(
            config_params=config_params,
            cli_params=cli_params,
            mode_params=mode_params,
        )

        try:
            instance = cls(**separated)
        except Exception as exc:  # pragma: no cover - covered via runtime errors
            raise DynVisionValidationError(
                f"{cls.__name__} creation failed: {exc}"
            ) from exc

        provenance_map = getattr(separated, "provenance", {})
        object.__setattr__(instance, "_value_provenance", provenance_map)
        object.__setattr__(instance, "_mode_resolution", mode_resolution)
        return instance

    @classmethod
    def _get_config_and_cli_params_separate(
        cls,
        config_path: Optional[str] = None,
        override_kwargs: Optional[Dict[str, Any]] = None,
        args: Optional[List[str]] = None,
    ) -> Tuple[ParamsDict, ParamsDict]:
        """Get config params and CLI params as separate dicts.

        Returns:
            Tuple of (config_params, cli_params)
        """
        config_params: Dict[str, Any] = {}
        config_sources: Dict[str, ProvenanceRecord] = {}
        cli_params: Dict[str, Any] = {}
        cli_sources: Dict[str, ProvenanceRecord] = {}

        # Parse CLI args first to extract config_path if present
        # Note: args=None means use sys.argv, which is the common case
        cli_args = cls._parse_cli_args(args)
        # Extract config_path from CLI if not provided as parameter
        if not config_path and "config_path" in cli_args:
            config_path = cli_args.pop("config_path")
        else:
            cli_args.pop("config_path", None)  # Remove if present
        cli_params.update(cli_args)
        for key in cli_args:
            cli_sources[key] = ProvenanceRecord(source="cli")
        logger.debug(f"Parsed {len(cli_args)} parameters from CLI")

        # Load config file (may come from parameter or extracted from CLI args)
        if config_path:
            config_file_params = cls._load_config_file(config_path)
            for key, value in config_file_params.items():
                config_params[key] = value
                config_sources[key] = ProvenanceRecord(source="config")
            logger.debug(f"Loaded {len(config_params)} parameters from config file")

        # Add override kwargs (highest priority within CLI)
        if override_kwargs:
            cli_params.update(override_kwargs)
            for key in override_kwargs:
                cli_sources[key] = ProvenanceRecord(source="override")
            logger.debug(f"Applied {len(override_kwargs)} direct overrides")

        # DO NOT resolve aliases here!
        # Alias resolution converts unscoped to scoped (e.g., model_name -> model.model_name)
        # which would incorrectly override explicitly scoped values within the same source.
        # Aliases are resolved in _separate_single_source after scoped/unscoped precedence.

        return ParamsDict(config_params, provenance=config_sources), ParamsDict(
            cli_params, provenance=cli_sources
        )

    @classmethod
    def _resolve_mode_overrides(
        cls, config_params: ParamsDict, cli_params: ParamsDict
    ) -> Tuple[Optional[ParamsDict], ModeResolution]:
        """Resolve activated mode patches and return them as a ParamsDict."""

        toggle_values = cls._gather_mode_toggle_values(config_params, cli_params)
        context = cls._build_mode_context(config_params, cli_params)
        resolution = ModeRegistry.resolve_modes(toggle_values, context)

        flattened, provenance = cls._flatten_mode_patches(resolution)
        if not flattened:
            return None, resolution

        return ParamsDict(flattened, provenance=provenance), resolution

    @classmethod
    def _strip_mode_toggle_keys(
        cls,
        config_params: ParamsDict,
        cli_params: ParamsDict,
    ) -> None:
        """Remove canonical and shortcut mode toggles from raw parameter dicts."""

        for mode_name in ModeRegistry.list_modes():
            definition = ModeRegistry.get_definition(mode_name)
            if not definition:
                continue
            shortcut_keys = (definition.toggle_key, mode_name)
            for key in shortcut_keys:
                if key in config_params:
                    config_params.pop(key, None)
                    config_params.provenance.pop(key, None)
                if key in cli_params:
                    cli_params.pop(key, None)
                    cli_params.provenance.pop(key, None)

    @classmethod
    def _gather_mode_toggle_values(
        cls, config_params: Mapping[str, Any], cli_params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Collect toggle values with CLI taking precedence over config."""

        toggle_values: Dict[str, Any] = {}
        config_dict = dict(config_params)
        cli_dict = dict(cli_params)
        for mode_name in ModeRegistry.list_modes():
            definition = ModeRegistry.get_definition(mode_name)
            if not definition:
                continue
            value = definition.default_toggle
            shortcut_keys = (definition.toggle_key, mode_name)
            for key in shortcut_keys:
                if key in config_dict:
                    value = config_dict[key]
            for key in shortcut_keys:
                if key in cli_dict:
                    value = cli_dict[key]
            toggle_values[definition.toggle_key] = value
        return toggle_values

    @staticmethod
    def _build_mode_context(
        config_params: Mapping[str, Any], cli_params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Build context dictionary combining config and CLI values."""

        context = dict(config_params)
        context.update(dict(cli_params))
        return context

    @classmethod
    def _flatten_mode_patches(
        cls, resolution: ModeResolution
    ) -> Tuple[Dict[str, Any], Dict[str, ProvenanceRecord]]:
        """Flatten active mode payloads into dotted keys with provenance."""

        flattened: Dict[str, Any] = {}
        provenance: Dict[str, ProvenanceRecord] = {}

        for mode_name in resolution.active_modes:
            payload = resolution.patches.get(mode_name, {})
            flattened_payload = cls._flatten_nested_payload(payload)
            for key, value in flattened_payload.items():
                flattened[key] = value
                provenance[key] = ProvenanceRecord(source=f"mode:{mode_name}")

        return flattened, provenance

    @staticmethod
    def _flatten_nested_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten arbitrarily nested dictionaries into dotted keys."""

        flattened: Dict[str, Any] = {}

        def _recurse(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    new_prefix = f"{prefix}.{subkey}" if prefix else subkey
                    _recurse(new_prefix, subvalue)
            else:
                flattened[prefix] = value

        for key, value in payload.items():
            _recurse(key, value)

        return flattened

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def persist_resolved_config(
        self,
        primary_output: Path | str,
        script_name: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Write the fully resolved parameter set alongside an output artifact."""

        target = Path(f"{primary_output}.config.yaml")
        target.parent.mkdir(parents=True, exist_ok=True)

        mode_resolution = getattr(
            self,
            "_mode_resolution",
            ModeResolution(
                toggles={}, raw_values={}, active_modes=tuple(), patches={}
            ),
        )

        flat_params = self._build_flat_parameter_map()
        flat_params["_active_modes"] = list(mode_resolution.active_modes)

        provenance_map = self._build_flat_provenance_map()
        if provenance_map:
            flat_params["_provenance"] = provenance_map

        metadata = {
            "generated_at": datetime.utcnow().isoformat(),
            "script": script_name,
            "primary_output": str(primary_output),
            "source_precedence": "config -> modes -> cli",
            "active_modes": ", ".join(mode_resolution.active_modes) or "none",
        }
        if mode_resolution.raw_values:
            metadata["mode_toggles"] = mode_resolution.raw_values
        if extra_metadata:
            metadata.update(extra_metadata)

        header_lines = ["# DynVision resolved configuration"] + [
            f"# {key.replace('_', ' ').title()}: {value}"
            for key, value in metadata.items()
        ]
        header = "\n".join(header_lines) + "\n\n"

        serializable_params = {
            key: self._prepare_yaml_value(value) for key, value in flat_params.items()
        }

        with target.open("w", encoding="utf-8") as handle:
            handle.write(header)
            yaml.safe_dump(
                serializable_params,
                handle,
                default_flow_style=False,
                sort_keys=False,
            )

        return target

    def _build_flat_parameter_map(self) -> Dict[str, Any]:
        """Flatten composite and component parameters into dotted keys."""

        payload = self.model_dump()
        component_names = set(self.get_component_classes().keys())
        flattened: Dict[str, Any] = {}

        for key, value in payload.items():
            if key in component_names and isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flattened[f"{key}.{subkey}"] = subvalue
            else:
                flattened[key] = value

        return flattened

    def _build_flat_provenance_map(self) -> Dict[str, str]:
        """Create a dotted-key provenance map suitable for YAML serialization."""

        flattened: Dict[str, str] = {}

        base_provenance = getattr(self, "_value_provenance", {})
        for key, record in base_provenance.items():
            flattened[key] = record.format() or "default"

        for comp_name, component in self.iter_components():
            comp_provenance = getattr(component, "_value_provenance", {})
            for key, record in comp_provenance.items():
                flattened[f"{comp_name}.{key}"] = record.format() or "default"

        return flattened

    def _prepare_yaml_value(self, value: Any) -> Any:
        """Recursively coerce values to YAML-safe primitives."""

        if isinstance(value, dict):
            return {k: self._prepare_yaml_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._prepare_yaml_value(v) for v in value]
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if torch is not None:
            if isinstance(value, torch.Tensor):  # Avoid device refs in YAML
                tensor = value.detach().cpu()
                if tensor.numel() == 1:
                    return tensor.item()
                return tensor.tolist()
            if isinstance(value, (torch.dtype, torch.device, torch.Size)):
                return str(value)
        return value

    @classmethod
    def _separate_component_configs_two_sources(
        cls,
        config_params: ParamsDict,
        cli_params: ParamsDict,
        mode_params: Optional[ParamsDict] = None,
    ) -> ParamsDict:
        """
        Split parameters from multiple sources with proper precedence.

        Within each source, apply scoped > unscoped precedence. Then merge sources
        in the order config → modes → CLI, with later sources overriding earlier ones.

        Args:
            config_params: Parameters from config files
            cli_params: Parameters from CLI arguments
            mode_params: Parameters injected by active modes (optional)

        Returns:
            Dictionary with instantiated components and composite base fields
        """
        source_specs = [
            ("config", config_params),
            ("modes", mode_params),
            ("cli", cli_params),
        ]

        separated_sources: List[
            Tuple[
                str, Dict[str, Dict[str, Any]], Dict[str, Dict[str, ProvenanceRecord]]
            ]
        ] = []

        for label, params in source_specs:
            if not params:
                continue
            source_map = getattr(params, "provenance", {})
            data = dict(params)
            if not data:
                continue
            components, provenance = cls._separate_single_source(data, source_map)
            separated_sources.append((label, components, provenance))

        component_classes = cls.get_component_classes()
        final_configs: Dict[str, Dict[str, Any]] = {}
        final_provenance: Dict[str, Dict[str, ProvenanceRecord]] = {}

        # Merge composite base fields honoring source order (last source wins)
        composite_base: Dict[str, Any] = {}
        composite_base_provenance: Dict[str, ProvenanceRecord] = {}
        for _, components, provenance in separated_sources:
            base_values = components.get("_composite_base", {})
            composite_base.update(base_values)
            composite_base_provenance.update(provenance.get("_composite_base", {}))

        for comp_name in component_classes:
            final_configs[comp_name] = composite_base.copy()
            final_provenance[comp_name] = {
                key: composite_base_provenance[key]
                for key in composite_base
                if key in composite_base_provenance
            }

        # Apply component overrides in source order
        for _, components, provenance in separated_sources:
            for comp_name in component_classes:
                comp_values = components.get(comp_name, {})
                if comp_values:
                    final_configs[comp_name].update(comp_values)
                if comp_name in provenance:
                    final_provenance[comp_name].update(provenance[comp_name])

        # Apply preprocessors
        preprocessors = cls.get_component_preprocessors()
        for comp_name, preprocessor in preprocessors.items():
            if comp_name in final_configs and preprocessor is not None:
                original_values = final_configs[comp_name].copy()
                updated = preprocessor(final_configs[comp_name])
                if updated is not None:
                    final_configs[comp_name] = updated
                else:
                    updated = final_configs[comp_name]

                provenance_map = final_provenance.setdefault(comp_name, {})
                for key, value in updated.items():
                    if key in original_values and original_values[key] == value:
                        continue
                    record = provenance_map.get(
                        key, ProvenanceRecord(source="default")
                    )
                    provenance_map[key] = record.add_mutation("derived")

        # Instantiate components
        instantiated: Dict[str, Any] = {}
        for comp_name, comp_cls in component_classes.items():
            try:
                component_instance = comp_cls(**final_configs[comp_name])
                object.__setattr__(
                    component_instance,
                    "_value_provenance",
                    final_provenance.get(comp_name, {}),
                )
                instantiated[comp_name] = component_instance
            except Exception as exc:
                raise DynVisionValidationError(
                    f"{cls.__name__} component '{comp_name}' validation failed: {exc}"
                ) from exc

        instantiated.update(composite_base)
        return ParamsDict(instantiated, provenance=composite_base_provenance)

    @classmethod
    def _separate_single_source(
        cls,
        params: Dict[str, Any],
        sources: Optional[Dict[str, ProvenanceRecord]] = None,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, ProvenanceRecord]]]:
        """
        Split parameters from a single source into component configs.

        Applies scoped > unscoped precedence within this source only and returns
        both the resolved parameter dictionary and the associated provenance map.
        """
        sources = sources or {}
        params, sources = cls._resolve_aliases_with_precedence(params, sources)

        component_classes = cls.get_component_classes()
        mode = cls._get_active_mode()

        component_field_sets = {
            name: set(comp.model_fields.keys())
            for name, comp in component_classes.items()
        }
        base_fields = set(cls.model_fields.keys()) - set(component_classes.keys())

        # Storage for parameters at each precedence level and their provenance
        level_5_base: Dict[str, Any] = {}
        level_5_base_prov: Dict[str, ProvenanceRecord] = {}
        level_4_component = {name: {} for name in component_classes}
        level_4_component_prov = {name: {} for name in component_classes}
        level_3_mode: Dict[str, Any] = {}
        level_3_mode_prov: Dict[str, ProvenanceRecord] = {}
        level_2_mode_component = {name: {} for name in component_classes}
        level_2_mode_component_prov = {name: {} for name in component_classes}
        composite_base: Dict[str, Any] = {}
        composite_base_prov: Dict[str, ProvenanceRecord] = {}

        def _record_for_key(key: str) -> ProvenanceRecord:
            return sources.get(key, ProvenanceRecord(source="default"))

        # ===== PHASE 1: Classify parameters by scope =====
        for key, value in params.items():
            parts = key.split(".")
            record = _record_for_key(key)

            if len(parts) == 3:
                mode_prefix, comp_name, param_name = parts
                if mode and mode_prefix == mode and comp_name in component_classes:
                    scoped_record = record.with_scope(f"{mode}.{comp_name}")
                    level_2_mode_component[comp_name][param_name] = value
                    level_2_mode_component_prov[comp_name][param_name] = scoped_record
                    logger.debug(
                        f"Mode+Component: {comp_name}.{param_name}={value} [mode={mode}]"
                    )
                    continue
                level_5_base[key] = value
                level_5_base_prov[key] = record
                continue

            if len(parts) == 2:
                prefix, param_name = parts
                if mode and prefix == mode:
                    scoped_record = record.with_scope(mode)
                    level_3_mode[param_name] = value
                    level_3_mode_prov[param_name] = scoped_record
                    logger.debug(f"Mode: {param_name}={value} [mode={mode}]")
                    continue

                if prefix in component_classes:
                    scoped_record = record.with_scope(prefix)
                    level_4_component[prefix][param_name] = value
                    level_4_component_prov[prefix][param_name] = scoped_record
                    logger.debug(f"Component: {prefix}.{param_name}={value}")
                    continue

                level_5_base[key] = value
                level_5_base_prov[key] = record
                continue

            if key in base_fields:
                composite_base[key] = value
                composite_base_prov[key] = record

            level_5_base[key] = value
            level_5_base_prov[key] = record

        # ===== PHASE 2: Apply precedence hierarchy within this source =====
        component_configs: Dict[str, Dict[str, Any]] = {}
        component_provenance: Dict[str, Dict[str, ProvenanceRecord]] = {
            name: {} for name in component_classes
        }

        explicitly_scoped = set()
        for comp_name in component_classes:
            for key in level_4_component[comp_name]:
                explicitly_scoped.add((comp_name, key))
            for key in level_2_mode_component[comp_name]:
                explicitly_scoped.add((comp_name, key))

        for comp_name in component_classes:
            comp_fields = component_field_sets[comp_name]
            comp_config: Dict[str, Any] = {}
            comp_prov: Dict[str, ProvenanceRecord] = {}

            for key, value in level_5_base.items():
                if key in comp_fields and (comp_name, key) not in explicitly_scoped:
                    comp_config[key] = value
                    comp_prov[key] = level_5_base_prov[key].with_scope(comp_name)
                    logger.debug(
                        f"Unscoped '{key}' routed to {comp_name} (no explicit scope in this source)"
                    )

            for key, value in level_4_component[comp_name].items():
                comp_config[key] = value
                comp_prov[key] = level_4_component_prov[comp_name][key]

            for key, value in level_3_mode.items():
                if key in comp_fields:
                    comp_config[key] = value
                    comp_prov[key] = level_3_mode_prov[key]

            for key, value in level_2_mode_component[comp_name].items():
                comp_config[key] = value
                comp_prov[key] = level_2_mode_component_prov[comp_name][key]

            component_configs[comp_name] = comp_config
            component_provenance[comp_name] = comp_prov

        # ===== PHASE 3: Handle remaining unscoped parameters =====
        for key, value in level_5_base.items():
            already_handled = any(
                key in component_field_sets[comp] for comp in component_classes
            )
            if already_handled:
                continue

            cls._handle_unscoped_param(key, value, component_configs, composite_base)
            if key not in composite_base_prov:
                composite_base_prov[key] = level_5_base_prov[key]
            for comp_name in component_classes:
                if (
                    key in component_configs.get(comp_name, {})
                    and key not in component_provenance[comp_name]
                ):
                    component_provenance[comp_name][key] = level_5_base_prov[
                        key
                    ].with_scope(comp_name)

        # ===== PHASE 4: Add composite base fields to all components =====
        for comp_name in component_configs:
            for base_key, base_value in composite_base.items():
                if base_key not in component_configs[comp_name]:
                    component_configs[comp_name][base_key] = base_value
                    base_record = composite_base_prov.get(
                        base_key, ProvenanceRecord(source="default")
                    )
                    component_provenance[comp_name][base_key] = base_record.with_scope(
                        comp_name
                    )

        component_configs["_composite_base"] = composite_base
        component_provenance["_composite_base"] = composite_base_prov

        return component_configs, component_provenance

    @classmethod
    def _separate_component_configs(cls, params: Dict[str, Any]) -> ParamsDict:
        """
        Split flat parameters into component configs (legacy single-source method).

        For backward compatibility with direct instantiation (not from_cli_and_config).
        Applies scoped > unscoped precedence within the single parameter dict.

        For proper config vs CLI precedence, use from_cli_and_config instead.
        """
        # Use single source separation
        component_configs, component_provenance = cls._separate_single_source(params)

        # Remove _composite_base marker and merge into result
        composite_base = component_configs.pop("_composite_base", {})
        composite_base_prov = component_provenance.pop("_composite_base", {})

        # Apply preprocessors
        component_classes = cls.get_component_classes()
        preprocessors = cls.get_component_preprocessors()
        for comp_name, preprocessor in preprocessors.items():
            if comp_name in component_configs and preprocessor is not None:
                component_configs[comp_name] = preprocessor(
                    component_configs[comp_name]
                )

        # Instantiate components
        instantiated: Dict[str, Any] = {}
        for comp_name, comp_cls in component_classes.items():
            try:
                component_instance = comp_cls(**component_configs[comp_name])
                object.__setattr__(
                    component_instance,
                    "_value_provenance",
                    component_provenance.get(comp_name, {}),
                )
                instantiated[comp_name] = component_instance
            except Exception as exc:
                raise DynVisionValidationError(
                    f"{cls.__name__} component '{comp_name}' validation failed: {exc}"
                ) from exc

        instantiated.update(composite_base)
        params_dict = ParamsDict(instantiated, provenance=composite_base_prov)
        return params_dict

    @staticmethod
    def _find_component_targets(
        key: str,
        component_field_sets: Dict[str, set],
        order: Iterable[str],
    ) -> Tuple[str, ...]:
        """Return component names whose schemas contain ``key`` in precedence order."""
        targets: List[str] = []
        for component_name in order:
            if key in component_field_sets.get(component_name, set()):
                targets.append(component_name)
        return tuple(targets)

    @classmethod
    def _assign_to_components(
        cls,
        key: str,
        value: Any,
        component_data: Dict[str, ComponentDict],
        targets: Tuple[str, ...],
    ) -> None:
        """Assign value to one or more component dictionaries."""
        for component_name in targets:
            component_data[component_name][key] = value

    @classmethod
    def _handle_unscoped_param(
        cls,
        key: str,
        value: Any,
        component_data: Dict[str, ComponentDict],
        base_params: Dict[str, Any],
    ) -> None:
        """Fallback handler for keys that do not match any component field."""
        logger.debug(
            "Unscoped parameter '%s' assigned to composite base namespace", key
        )
        base_params[key] = value
