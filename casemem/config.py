from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Models:
    backbone: str = "Qwen/Qwen3-VL-8B-Instruct"
    teacher: str = "qwen3.5-flash"
    text_encoder: str = "BAAI/bge-m3"
    visual_encoder: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    detector: str = "IDEA-Research/grounding-dino-base"


@dataclass(frozen=True)
class Memory:
    merge_threshold: float = 0.9
    max_teacher_regions: int = 3
    detector_box_threshold: float = 0.25
    detector_text_threshold: float = 0.25
    guide_descriptions: int = 4

    def __post_init__(self):
        for value in (self.merge_threshold, self.detector_box_threshold, self.detector_text_threshold):
            if not 0 <= value <= 1:
                raise ValueError("Memory thresholds must be in [0, 1]")
        if min(self.max_teacher_regions, self.guide_descriptions) < 1:
            raise ValueError("Candidate and description limits must be positive")


@dataclass(frozen=True)
class Retrieval:
    text_top_k: int = 5
    visual_top_k: int = 3
    alpha: float = 0.5
    grid_size: int = 512
    patch_size: int = 128
    stride: int = 64
    sigma: float = 0.20
    spatial_weight: float = 0.5
    score_threshold: float = 1.2
    align_threshold: float = 0.8
    cosine_threshold: float = 0.7
    epsilon: float = 1e-8

    def __post_init__(self):
        if min(self.text_top_k, self.visual_top_k, self.stride, self.patch_size) < 1:
            raise ValueError("Top-k values, patch size and stride must be positive")
        if self.patch_size > self.grid_size or self.stride > self.patch_size:
            raise ValueError("Patch size must fit the grid and stride must not leave gaps")
        if not 0 <= self.alpha <= 1 or not 0 <= self.align_threshold <= 1:
            raise ValueError("alpha and align_threshold must be in [0, 1]")
        if not -1 <= self.cosine_threshold <= 1:
            raise ValueError("cosine_threshold must be in [-1, 1]")
        for value in (self.sigma, self.epsilon):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("sigma and epsilon must be finite and positive")
        if not math.isfinite(self.score_threshold) or not math.isfinite(self.spatial_weight) or self.spatial_weight < 0:
            raise ValueError("Invalid combined score parameters")


@dataclass(frozen=True)
class Scoring:
    open_metric: str = "exact"
    open_correct_threshold: float = 1.0

    def __post_init__(self):
        if self.open_metric not in {"exact", "token_f1"} or not 0 < self.open_correct_threshold <= 1:
            raise ValueError("Use exact or token_f1 and a correctness threshold in (0, 1]")


@dataclass(frozen=True)
class Training:
    steps: int = 200
    queries_per_batch: int = 16
    rollouts_per_query: int = 8
    localization_weight: float = 0.2
    learning_rate: float = 1e-6
    prompt_tokens: int = 4096
    response_tokens: int = 2048

    def __post_init__(self):
        if min(self.steps, self.queries_per_batch, self.prompt_tokens, self.response_tokens) < 1 or self.rollouts_per_query < 2:
            raise ValueError("Invalid GRPO budget")
        if not math.isfinite(self.localization_weight) or self.localization_weight < 0:
            raise ValueError("Invalid localization reward weight")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Invalid learning rate")


@dataclass(frozen=True)
class Inference:
    batch_size: int = 64
    max_sequence_tokens: int = 2048
    max_new_tokens: int = 512

    def __post_init__(self):
        if self.batch_size < 1 or not 0 < self.max_new_tokens < self.max_sequence_tokens:
            raise ValueError("Invalid inference batch size or sequence limit")


@dataclass(frozen=True)
class Config:
    models: Models = field(default_factory=Models)
    memory: Memory = field(default_factory=Memory)
    retrieval: Retrieval = field(default_factory=Retrieval)
    scoring: Scoring = field(default_factory=Scoring)
    training: Training = field(default_factory=Training)
    inference: Inference = field(default_factory=Inference)


def load_config(path: str | None) -> Config:
    if path is None:
        return Config()
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = {item.name: type(getattr(Config(), item.name)) for item in fields(Config)}
    unexpected = values.keys() - classes.keys()
    if unexpected:
        raise ValueError(f"Unknown config sections: {sorted(unexpected)}")
    return Config(**{key: classes[key](**section) for key, section in values.items()})


def config_dict(config: Config) -> dict:
    return asdict(config)
