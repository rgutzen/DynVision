"""Tests for mode registry heuristics and config persistence."""

from __future__ import annotations

import yaml

from dynvision.params import TrainingParams
from dynvision.params.mode_registry import ModeRegistry
from tests.params.utils import create_temp_config, temporary_training_paths


def _load_yaml_body(path):
    content = path.read_text()
    body_lines = [line for line in content.splitlines() if not line.startswith("#")]
    return yaml.safe_load("\n".join(body_lines))


def test_persist_resolved_config_writes_metadata(tmp_path):
    config_file = create_temp_config({"debug": True})
    try:
        with temporary_training_paths() as training_paths:
            params = TrainingParams.from_cli_and_config(
                config_path=str(config_file),
                override_kwargs={**training_paths, "debug": True},
            )

        output_model = tmp_path / "model.pt"
        output_model.write_bytes(b"")
        persisted_path = params.persist_resolved_config(
            primary_output=output_model,
            script_name="tests/params/test_mode_registry_and_persistence.py",
        )

        assert persisted_path.exists()
        data = _load_yaml_body(persisted_path)

        assert data["trainer.enable_progress_bar"] is True
        assert data["trainer.log_every_n_steps"] == 1
        assert data["_active_modes"] == ["debug"]
        provenance = data.get("_provenance", {})
        assert "trainer.enable_progress_bar" in provenance
        assert "mode:debug" in provenance["trainer.enable_progress_bar"]
    finally:
        config_file.unlink(missing_ok=True)


def test_mode_registry_resolves_debug_auto():
    ModeRegistry.reload()
    toggle_source = {"use_debug_mode": "auto"}
    context = {"log_level": "DEBUG", "epochs": 100}
    resolution = ModeRegistry.resolve_modes(toggle_source, context)
    assert "debug" in resolution.active_modes
    assert resolution.toggles["debug"] is True
    assert "debug" in resolution.patches


def test_mode_registry_resolves_large_dataset_auto():
    ModeRegistry.reload()
    toggle_source = {"use_large_dataset_mode": "auto"}
    context = {"data_name": "imagenet", "local_execution": True}
    resolution = ModeRegistry.resolve_modes(toggle_source, context)
    assert "large_dataset" in resolution.active_modes
    assert resolution.toggles["large_dataset"] is True
    assert resolution.patches["large_dataset"]
