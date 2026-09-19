"""Central experiment configuration.

Do not alter scientific settings here without documenting the reason in the
README and in a run's metadata file.
"""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str = "Qwen/Qwen3-1.7B"
    dataset_name: str = "tau/commonsense_qa"
    dataset_split: str = "validation"
    final_sample_size: int = 300
    pilot_sample_size: int = 10
    seed: int = 42
    max_new_tokens: int = 64
    do_sample: bool = False
    enable_thinking: bool = False
    device_map: str = "auto"
    output_dir: str = "outputs"

    def sample_size(self, pilot: bool) -> int:
        return self.pilot_sample_size if pilot else self.final_sample_size

    def output_path(self, pilot: bool) -> Path:
        mode = "pilot" if pilot else "final"
        return Path(self.output_dir) / mode

    def as_dict(self) -> dict:
        return asdict(self)
