from dataclasses import dataclass, field
from io import BytesIO
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from casemem.scoring import valid_box


def unit_vector(vector):
    array = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    return array / norm if np.isfinite(norm) and norm > 0 else np.zeros_like(array)


@dataclass
class Instance:
    source_id: str
    image_hash: str
    box: tuple[float, float, float, float]
    concept: str
    full_feature: np.ndarray
    crop_feature: np.ndarray
    crop_png: bytes


@dataclass
class Cluster:
    descriptions: list[str]
    feature: np.ndarray
    instances: list[Instance] = field(default_factory=list)


@dataclass
class Bank:
    text_encoder: str
    visual_encoder: str
    build_id: str = ""
    clusters: list[Cluster] = field(default_factory=list)
    completed: set[str] = field(default_factory=set)
    stats: dict = field(default_factory=dict)

    def add(self, description, instance, encoder, threshold, max_descriptions):
        description = " ".join(description.split())
        text_feature = unit_vector(encoder.text([description])[0])
        if not np.any(text_feature):
            raise ValueError("The text encoder returned a zero embedding for a memory description")
        if self.clusters:
            scores = np.stack([cl.feature for cl in self.clusters]) @ text_feature
            best = int(np.argmax(scores))
            if scores[best] > threshold:
                cluster = self.clusters[best]
                cluster.instances.append(instance)
                if description not in cluster.descriptions and len(cluster.descriptions) < max_descriptions:
                    cluster.descriptions.append(description)
                    cluster.feature = unit_vector(encoder.text([". ".join(cluster.descriptions)])[0])
                return
        self.clusters.append(Cluster([description], text_feature, [instance]))

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": 1, "text_encoder": self.text_encoder, "visual_encoder": self.visual_encoder,
            "build_id": self.build_id, "completed": sorted(self.completed),
            "stats": self.stats, "clusters": [],
        }
        entries = [item for cluster in self.clusters for item in cluster.instances]
        vectors = {
            "guides": _stack(cluster.feature for cluster in self.clusters),
            "full": _stack(item.full_feature for item in entries),
            "crop": _stack(item.crop_feature for item in entries),
        }
        with BytesIO() as arrays:
            np.savez(arrays, **vectors)
            with NamedTemporaryFile(dir=target.parent, delete=False) as temp:
                temp_path = Path(temp.name)
            try:
                with ZipFile(temp_path, "w", compression=ZIP_DEFLATED) as archive:
                    index = 0
                    for cluster in self.clusters:
                        members = []
                        for item in cluster.instances:
                            filename = f"crops/{index:08d}.png"
                            archive.writestr(filename, item.crop_png)
                            members.append({"source_id": item.source_id, "image_hash": item.image_hash,
                                            "box": item.box, "concept": item.concept, "crop": filename})
                            index += 1
                        meta["clusters"].append({"descriptions": cluster.descriptions, "instances": members})
                    archive.writestr("metadata.json", json.dumps(meta, ensure_ascii=False, allow_nan=False))
                    archive.writestr("embeddings.npz", arrays.getvalue())
                os.replace(temp_path, target)
            finally:
                temp_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with ZipFile(path) as archive:
            meta = json.loads(archive.read("metadata.json"))
            if meta["version"] != 1:
                raise ValueError(f"Unsupported bank format version: {meta['version']}")
            bank = cls(meta["text_encoder"], meta["visual_encoder"], meta["build_id"],
                       completed=set(meta["completed"]), stats=meta["stats"])
            with np.load(BytesIO(archive.read("embeddings.npz")), allow_pickle=False) as vectors:
                guides, full, crops = vectors["guides"], vectors["full"], vectors["crop"]
            count = sum(len(cluster["instances"]) for cluster in meta["clusters"])
            if len(guides) != len(meta["clusters"]) or len(full) != count or len(crops) != count:
                raise ValueError("Bank metadata and feature counts disagree")
            if any(not np.all(np.isfinite(values)) for values in (guides, full, crops)):
                raise ValueError("The bank contains non-finite embeddings")
            index = 0
            for cluster, feature in zip(meta["clusters"], guides, strict=True):
                instances = []
                for item in cluster["instances"]:
                    box = valid_box(item["box"])
                    if box is None:
                        raise ValueError("The bank contains an invalid normalized box")
                    instances.append(Instance(item["source_id"], item["image_hash"], box, item["concept"],
                                              full[index], crops[index], archive.read(item["crop"])))
                    index += 1
                bank.clusters.append(Cluster(cluster["descriptions"], feature, instances))
        return bank

    def check_encoders(self, models):
        if (self.text_encoder, self.visual_encoder) != (models.text_encoder, models.visual_encoder):
            raise ValueError("The bank was built with different encoders. Use its matching configuration.")


def _stack(vectors):
    values = list(vectors)
    return np.stack(values).astype(np.float32) if values else np.empty((0, 0), dtype=np.float32)
