"""Shared utilities for composite parameter classes."""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Dict, Iterable, List, Optional, Tuple, Type
import logging

from dynvision.params.base_params import BaseParams, DynVisionValidationError

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
            mode_overrides = config.pop(mode)  # Remove mode section from config
            if isinstance(mode_overrides, dict):
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
        cls, params: Dict[str, Any]
    ) -> Dict[str, Any]:
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
            return params

        resolved = params.copy()

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
                    # Remove alias in both cases
                    del resolved[alias]
                else:
                    # Only alias exists - resolve it to target
                    logger.debug(
                        f"Cross-scope alias '{alias}'={alias_value} resolves to '{target}' "
                        f"(alias scope {alias_scope}, target scope {target_scope})"
                    )
                    resolved[target] = alias_value
                    del resolved[alias]

        return resolved

        return resolved

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

        # Separate each source independently, then merge with CLI taking precedence
        separated = cls._separate_component_configs_two_sources(
            config_params=config_params, cli_params=cli_params
        )

        try:
            return cls(**separated)
        except Exception as exc:  # pragma: no cover - covered via runtime errors
            raise DynVisionValidationError(
                f"{cls.__name__} creation failed: {exc}"
            ) from exc

    @classmethod
    def _get_config_and_cli_params_separate(
        cls,
        config_path: Optional[str] = None,
        override_kwargs: Optional[Dict[str, Any]] = None,
        args: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Get config params and CLI params as separate dicts.

        Returns:
            Tuple of (config_params, cli_params)
        """
        config_params = {}
        cli_params = {}

        # Parse CLI args first to extract config_path if present
        # Note: args=None means use sys.argv, which is the common case
        cli_args = cls._parse_cli_args(args)
        # Extract config_path from CLI if not provided as parameter
        if not config_path and "config_path" in cli_args:
            config_path = cli_args.pop("config_path")
        else:
            cli_args.pop("config_path", None)  # Remove if present
        cli_params.update(cli_args)
        logger.debug(f"Parsed {len(cli_args)} parameters from CLI")

        # Load config file (may come from parameter or extracted from CLI args)
        if config_path:
            config_params = cls._load_config_file(config_path)
            logger.debug(f"Loaded {len(config_params)} parameters from config file")

        # Add override kwargs (highest priority within CLI)
        if override_kwargs:
            cli_params.update(override_kwargs)
            logger.debug(f"Applied {len(override_kwargs)} direct overrides")

        # DO NOT resolve aliases here!
        # Alias resolution converts unscoped to scoped (e.g., model_name -> model.model_name)
        # which would incorrectly override explicitly scoped values within the same source.
        # Aliases are resolved in _separate_single_source after scoped/unscoped precedence.

        return config_params, cli_params

    @classmethod
    def _separate_component_configs_two_sources(
        cls, config_params: Dict[str, Any], cli_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Split parameters from two sources with proper precedence.

        Within each source (config and CLI), apply scoped > unscoped precedence.
        Then merge with ALL CLI values taking precedence over ALL config values.

        Args:
            config_params: Parameters from config files
            cli_params: Parameters from CLI arguments

        Returns:
            Dictionary with instantiated components and composite base fields
        """
        # Separate config params (scoped > unscoped within config)
        config_components = cls._separate_single_source(config_params)

        # Separate CLI params (scoped > unscoped within CLI)
        cli_components = cls._separate_single_source(cli_params)

        # Merge: CLI always wins over config
        component_classes = cls.get_component_classes()
        final_configs = {}

        # Merge composite base fields (CLI wins)
        composite_base = config_components.get("_composite_base", {}).copy()
        composite_base.update(cli_components.get("_composite_base", {}))

        for comp_name in component_classes:
            # Start with composite base (lowest priority)
            final_configs[comp_name] = composite_base.copy()
            # Add config values
            final_configs[comp_name].update(config_components.get(comp_name, {}))
            # Override with CLI values (all CLI beats all config)
            final_configs[comp_name].update(cli_components.get(comp_name, {}))

        # Apply preprocessors
        preprocessors = cls.get_component_preprocessors()
        for comp_name, preprocessor in preprocessors.items():
            if comp_name in final_configs and preprocessor is not None:
                final_configs[comp_name] = preprocessor(final_configs[comp_name])

        # Instantiate components
        instantiated = {}
        for comp_name, comp_cls in component_classes.items():
            try:
                instantiated[comp_name] = comp_cls(**final_configs[comp_name])
            except Exception as exc:
                raise DynVisionValidationError(
                    f"{cls.__name__} component '{comp_name}' validation failed: {exc}"
                ) from exc

        instantiated.update(composite_base)
        return instantiated

    @classmethod
    def _separate_single_source(
        cls, params: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Split parameters from a single source into component configs.

        Applies scoped > unscoped precedence within this source only.

        Handles aliases by converting them to their targets ONLY if the target
        doesn't already exist (scoped values beat aliases).

        Returns:
            Dictionary with component configs and '_composite_base' key for base fields
        """
        # Resolve aliases, but scoped params beat aliases
        params = cls._resolve_aliases_with_precedence(params)

        component_classes = cls.get_component_classes()
        mode = cls._get_active_mode()

        # Get field sets for routing
        component_field_sets = {
            name: set(comp.model_fields.keys())
            for name, comp in component_classes.items()
        }
        base_fields = set(cls.model_fields.keys()) - set(component_classes.keys())

        # Storage for parameters at each precedence level
        level_5_base = {}  # Unscoped
        level_4_component = {name: {} for name in component_classes}  # component.param
        level_3_mode = {}  # mode.param
        level_2_mode_component = {
            name: {} for name in component_classes
        }  # mode.component.param
        composite_base = {}  # Composite class fields

        # ===== PHASE 1: Classify parameters by scope =====
        for key, value in params.items():
            parts = key.split(".")

            # Check for mode.component.param (Level 2)
            if len(parts) == 3:
                mode_prefix, comp_name, param_name = parts
                if mode and mode_prefix == mode and comp_name in component_classes:
                    level_2_mode_component[comp_name][param_name] = value
                    logger.debug(
                        f"Mode+Component: {comp_name}.{param_name}={value} [mode={mode}]"
                    )
                    continue
                # Not our mode, treat as base
                level_5_base[key] = value
                continue

            # Check for mode.param (Level 3) or component.param (Level 4)
            if len(parts) == 2:
                prefix, param_name = parts

                # Is it our mode?
                if mode and prefix == mode:
                    level_3_mode[param_name] = value
                    logger.debug(f"Mode: {param_name}={value} [mode={mode}]")
                    continue

                # Is it a component?
                if prefix in component_classes:
                    level_4_component[prefix][param_name] = value
                    logger.debug(f"Component: {prefix}.{param_name}={value}")
                    continue

                # Neither mode nor component, treat as base
                level_5_base[key] = value
                continue

            # Single-part key (Level 5 or composite base)
            # Note: A key can be BOTH in base_fields AND in component fields
            # (e.g., seed, log_level are in InitParams AND in ModelParams/DataParams)
            if key in base_fields:
                composite_base[key] = value

            # Also add to level_5_base for component routing
            # This handles fields that exist in both composite and component classes
            level_5_base[key] = value

        # ===== PHASE 2: Apply precedence hierarchy within this source =====
        component_configs = {}

        # Track which parameters have explicit scoped values in THIS source
        explicitly_scoped = set()
        for comp_name in component_classes:
            for key in level_4_component[comp_name]:
                explicitly_scoped.add((comp_name, key))
            for key in level_2_mode_component[comp_name]:
                explicitly_scoped.add((comp_name, key))

        for comp_name in component_classes:
            comp_fields = component_field_sets[comp_name]
            comp_config = {}

            # Level 5: Base parameters (lowest priority)
            # Only use unscoped if not explicitly scoped at higher level IN THIS SOURCE
            for key, value in level_5_base.items():
                if key in comp_fields and (comp_name, key) not in explicitly_scoped:
                    comp_config[key] = value
                    logger.debug(
                        f"Unscoped '{key}' routed to {comp_name} (no explicit scope in this source)"
                    )

            # Level 4: Component-scoped
            comp_config.update(level_4_component[comp_name])

            # Level 3: Mode-scoped
            for key, value in level_3_mode.items():
                if key in comp_fields:
                    comp_config[key] = value

            # Level 2: Mode+Component-scoped (highest priority within this source)
            comp_config.update(level_2_mode_component[comp_name])

            component_configs[comp_name] = comp_config

        # ===== PHASE 3: Handle remaining unscoped parameters =====
        for key, value in level_5_base.items():
            # Skip if already handled in Phase 2
            already_handled = False
            for comp_name in component_classes:
                if key in component_field_sets[comp_name]:
                    already_handled = True
                    break

            if already_handled:
                continue

            # Parameter doesn't match any component field
            cls._handle_unscoped_param(key, value, component_configs, composite_base)

        # ===== PHASE 4: Add composite base fields to all components =====
        for comp_name in component_configs:
            for base_key, base_value in composite_base.items():
                if base_key not in component_configs[comp_name]:
                    component_configs[comp_name][base_key] = base_value

        # Store composite base separately for merging
        component_configs["_composite_base"] = composite_base

        return component_configs

    @classmethod
    def _separate_component_configs(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Split flat parameters into component configs (legacy single-source method).

        For backward compatibility with direct instantiation (not from_cli_and_config).
        Applies scoped > unscoped precedence within the single parameter dict.

        For proper config vs CLI precedence, use from_cli_and_config instead.
        """
        # Use single source separation
        component_configs = cls._separate_single_source(params)

        # Remove _composite_base marker and merge into result
        composite_base = component_configs.pop("_composite_base", {})

        # Apply preprocessors
        component_classes = cls.get_component_classes()
        preprocessors = cls.get_component_preprocessors()
        for comp_name, preprocessor in preprocessors.items():
            if comp_name in component_configs and preprocessor is not None:
                component_configs[comp_name] = preprocessor(
                    component_configs[comp_name]
                )

        # Instantiate components
        instantiated = {}
        for comp_name, comp_cls in component_classes.items():
            try:
                instantiated[comp_name] = comp_cls(**component_configs[comp_name])
            except Exception as exc:
                raise DynVisionValidationError(
                    f"{cls.__name__} component '{comp_name}' validation failed: {exc}"
                ) from exc

        instantiated.update(composite_base)
        return instantiated

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
