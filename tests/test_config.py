from src.config import ExperimentConfig


def test_final_and_pilot_sizes_and_output_paths_are_isolated():
    config = ExperimentConfig()
    assert config.sample_size(pilot=True) == 10
    assert config.sample_size(pilot=False) == 300
    assert config.sample_size(pilot=False) * 5 == 1500
    assert config.output_path(pilot=True).as_posix() == "outputs/pilot"
    assert config.output_path(pilot=False).as_posix() == "outputs/final"


def test_frozen_generation_settings():
    config = ExperimentConfig()
    assert config.max_new_tokens == 64
    assert config.do_sample is False
    assert config.enable_thinking is False
