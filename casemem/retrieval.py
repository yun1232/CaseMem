from dataclasses import dataclass
from functools import cached_property
from math import ceil, floor

import numpy as np

from casemem.memory import Instance, unit_vector


STRATEGIES = ("full", "spatial", "cosine")


@dataclass(frozen=True)
class Hit:
    instance: Instance
    score: float


@dataclass(frozen=True)
class Selection:
    hits: tuple[Hit, ...] = ()
    active: bool = False
    pixel_box: tuple[int, int, int, int] | None = None
    focus_box: tuple[float, float, float, float] | None = None
    score: float | None = None
    spatial: float | None = None
    visual: float | None = None


def pixel_box(box, width, height):
    x0, y0, x1, y1 = box
    return (max(0, floor(x0 * width)), max(0, floor(y0 * height)),
            min(width, ceil(x1 * width)), min(height, ceil(y1 * height)))


def normalized_box(box, width, height):
    return tuple(value / scale for value, scale in zip(box, (width, height, width, height), strict=True))


def mapped_cosine(dot):
    return np.clip((1.0 + dot) / 2.0, 0.0, 1.0)


def _offsets(size, patch, stride):
    last = size - patch
    positions = list(range(0, last + 1, stride))
    if positions[-1] != last:
        positions.append(last)
    return positions


class Retriever:
    def __init__(self, bank, encoder, settings):
        self.bank = bank
        self.encoder = encoder
        self.settings = settings

    @cached_property
    def guides(self):
        return np.stack([cluster.feature for cluster in self.bank.clusters])

    @cached_property
    def image_features(self):
        return [
            np.stack([unit_vector(item.full_feature) for item in cluster.instances])
            for cluster in self.bank.clusters
        ]

    @cached_property
    def grid_boxes(self):
        s = self.settings
        offsets = _offsets(s.grid_size, s.patch_size, s.stride)
        return np.asarray([(x, y, x + s.patch_size, y + s.patch_size)
                           for y in offsets for x in offsets], dtype=np.int64)

    @cached_property
    def grid_coordinates(self):
        return (np.arange(self.settings.grid_size, dtype=np.float64) + 0.5) / self.settings.grid_size

    def retrieve(self, image, question, *, source_hash=None):
        if not self.bank.clusters:
            return ()
        s = self.settings
        text_feature = unit_vector(self.encoder.text([question])[0])
        full_feature = unit_vector(self.encoder.images([image])[0])
        if not np.any(text_feature) or not np.any(full_feature):
            return ()
        text_scores = mapped_cosine(self.guides @ text_feature)
        guide_indices = np.argsort(-text_scores, kind="stable")[:s.text_top_k]
        candidates = []
        for index in guide_indices:
            image_scores = mapped_cosine(self.image_features[index] @ full_feature)
            scores = text_scores[index] ** s.alpha * image_scores ** (1.0 - s.alpha)
            guide = self.bank.clusters[index]
            for item, score in zip(guide.instances, scores, strict=True):
                if item.image_hash != source_hash and score > 0:
                    candidates.append(Hit(item, float(score)))
        candidates.sort(key=lambda hit: -hit.score)
        return tuple(candidates[:s.visual_top_k])

    def spatial_scores(self, hits):
        s = self.settings
        prior = np.zeros((s.grid_size, s.grid_size), dtype=np.float64)
        axis = self.grid_coordinates
        for hit in hits:
            x0, y0, x1, y1 = hit.instance.box
            x_kernel = np.exp(-0.5 * ((axis - (x0 + x1) / 2.0) / s.sigma) ** 2)
            y_kernel = np.exp(-0.5 * ((axis - (y0 + y1) / 2.0) / s.sigma) ** 2)
            prior += hit.score * np.outer(y_kernel, x_kernel)
        prior /= sum(hit.score for hit in hits) + s.epsilon
        integral = np.pad(prior.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
        x0, y0, x1, y1 = self.grid_boxes.T
        return (integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]) / s.patch_size ** 2

    def select(self, image, question, *, strategy="full", source_hash=None):
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
        hits = self.retrieve(image, question, source_hash=source_hash)
        if not hits:
            return Selection()
        s = self.settings
        spatial = self.spatial_scores(hits)
        visual = None
        if strategy != "spatial":
            raw_prototype = sum(hit.score * hit.instance.crop_feature for hit in hits) / (
                sum(hit.score for hit in hits) + s.epsilon
            )
            prototype = unit_vector(raw_prototype)
            if not np.any(prototype):
                return Selection(hits=hits)
            local = image.resize((s.grid_size, s.grid_size))
            patches = [local.crop(tuple(box)) for box in self.grid_boxes]
            features = np.asarray(self.encoder.images(patches), dtype=np.float32)
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            unit_patches = np.divide(features, norms, out=np.zeros_like(features), where=(norms > 0) & np.isfinite(norms))
            visual = np.clip(unit_patches @ prototype, -1.0, 1.0)
        if strategy == "full":
            combined = visual + s.spatial_weight * spatial
        elif strategy == "spatial":
            combined = spatial
        else:
            combined = visual
        index = int(np.argmax(combined))
        score = float(combined[index])
        align = float(spatial[index])
        if strategy == "full":
            active = score > s.score_threshold and align > s.align_threshold
        elif strategy == "spatial":
            active = score > s.align_threshold
        else:
            active = score > s.cosine_threshold
        crop = pixel_box(self.grid_boxes[index] / s.grid_size, image.width, image.height)
        return Selection(
            hits=hits, active=active, pixel_box=crop,
            focus_box=normalized_box(crop, image.width, image.height),
            score=score, spatial=align, visual=None if visual is None else float(visual[index]),
        )
