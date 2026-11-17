from collections.abc import Iterable
from dataclasses import dataclass, replace
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image, ImageOps


@dataclass(frozen=True)
class Example:
    id: str
    dataset: str
    image: Path
    question: str
    answers: tuple[str, ...]
    answer_type: str = "open"
    split: str | None = None

    def row(self) -> dict:
        return {
            "id": self.id, "dataset": self.dataset, "image": str(self.image),
            "question": self.question, "answers": list(self.answers),
            "answer_type": self.answer_type, "split": self.split,
        }


def load_examples(path: str | Path) -> list[Example]:
    manifest = Path(path).resolve()
    seen = set()
    examples = []
    with manifest.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                raw = row.get("answers", row.get("answer"))
                raw = raw if isinstance(raw, list) else [raw]
                answers = tuple(item.strip() for item in raw if isinstance(item, str) and item.strip())
                kind = str(row.get("answer_type", "open")).lower()
                kind = {"free-form": "open", "yes/no": "closed", "multiple_choice": "choice"}.get(kind, kind)
                image = (manifest.parent / row["image"]).resolve()
                source = str(row.get("dataset", "medvqa")).strip()
                if not source or not answers or kind not in {"open", "closed", "choice"}:
                    raise ValueError("Expected dataset, non-empty answer strings, and answer_type open/closed/choice")
                identifier = str(row.get("id", f"{source}:{line_no}"))
                question = row["question"].strip()
                if not identifier or not question or identifier in seen:
                    raise ValueError(f"Empty question/ID or duplicate ID: {identifier}")
                if not image.is_file():
                    raise FileNotFoundError(image)
                split = row.get("split")
                if split not in {None, "train", "validation", "test"}:
                    raise ValueError(f"Unknown split: {split}")
                seen.add(identifier)
                examples.append(Example(identifier, source, image, question, answers, kind, split))
            except (KeyError, TypeError, AttributeError, ValueError, OSError) as exc:
                raise ValueError(f"{manifest}:{line_no}: {exc}") from exc
    if not examples:
        raise ValueError(f"No examples in {manifest}")
    return examples


def open_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def image_digest(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    fingerprint = sha256(f"{rgb.width}:{rgb.height}:RGB:".encode())
    fingerprint.update(rgb.tobytes())
    return fingerprint.hexdigest()


def path_digests(examples: Iterable[Example]) -> dict[Path, str]:
    paths = dict.fromkeys(example.image for example in examples)
    return {path: image_digest(open_image(path)) for path in paths}


def png_bytes(image: Image.Image) -> bytes:
    with BytesIO() as stream:
        image.save(stream, format="PNG")
        return stream.getvalue()


def write_jsonl(path: str | Path, rows: Iterable[dict]):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as stream:
        try:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.replace(stream.name, target)
        finally:
            Path(stream.name).unlink(missing_ok=True)


def prepare_splits(train, validation, tests, *, validation_fraction=0.1, seed=0):
    for split, rows in (("train", train), ("validation", validation or []), ("test", tests)):
        if any(row.split not in {None, split} for row in rows):
            raise ValueError(f"Input manifest for {split} contains another split")
    digests = path_digests([*train, *(validation or []), *tests])
    test_keys = {digests[row.image] for row in tests}
    if validation is None:
        if not 0 < validation_fraction < 1:
            raise ValueError("The validation fraction must be in (0, 1)")
        eligible = [row for row in train if digests[row.image] not in test_keys]
        groups = set(digests[row.image] for row in eligible)
        if len(groups) < 2:
            raise ValueError("At least two training images are needed for a held-out validation split")
        ordered = sorted(groups, key=lambda key: sha256(f"{seed}:{key}".encode()).digest())
        count = min(max(1, round(len(ordered) * validation_fraction)), len(ordered) - 1)
        held_out = set(ordered[:count])
        validation = [row for row in eligible if digests[row.image] in held_out]
    clean_validation = [
        replace(row, split="validation") for row in validation if digests[row.image] not in test_keys
    ]
    excluded = test_keys | {digests[row.image] for row in clean_validation}
    clean_train = [replace(row, split="train") for row in train if digests[row.image] not in excluded]
    if not clean_train or not clean_validation:
        raise ValueError("Image-disjoint filtering leaves an empty training or validation split")
    return clean_train, clean_validation


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]
