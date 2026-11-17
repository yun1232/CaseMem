from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

from casemem.data import image_digest, write_jsonl
from casemem.scoring import normalize_answer


def import_huggingface(args):
    from datasets import load_dataset

    if args.parquet:
        collection = load_dataset("parquet", data_files={args.hf_split: args.parquet},
                                  split=args.hf_split, streaming=True)
    else:
        collection = load_dataset(args.dataset_id, name=args.subset, split=args.hf_split, streaming=True)
    output = Path(args.output_dir).resolve()
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    seen = set()

    def convert():
        for index, row in enumerate(collection):
            if args.language and row.get(args.language_key) != args.language:
                continue
            image = row[args.image_key]
            if isinstance(image, dict) and image.get("bytes"):
                with Image.open(BytesIO(image["bytes"])) as decoded:
                    image = ImageOps.exif_transpose(decoded).convert("RGB")
            if not isinstance(image, Image.Image):
                raise ValueError(f"Row {index} does not contain one decoded image in {args.image_key!r}")
            image = ImageOps.exif_transpose(image).convert("RGB")
            hashed = image_digest(image)
            image_path = image_dir / f"{hashed}.png"
            if not image_path.exists():
                image.save(image_path)
            answers = row[args.answer_key]
            answers = answers if isinstance(answers, list) else [answers]
            if not answers or any(not isinstance(answer, str) or not answer.strip() for answer in answers):
                raise ValueError(f"Row {index} has no usable reference answers")
            source_id = row[args.id_key] if args.id_key else index
            identifier = f"{args.dataset_name}:{args.hf_split}:{source_id}"
            if identifier in seen:
                raise ValueError(f"Duplicate input ID: {identifier}")
            seen.add(identifier)
            answer_type = str(row.get(args.type_key, "")).lower() if args.type_key else ""
            if answer_type not in {"open", "closed", "choice"}:
                binary = all(normalize_answer(answer) in {"yes", "no"} for answer in answers)
                answer_type = "closed" if binary else "open"
            yield {
                "id": identifier, "dataset": args.dataset_name, "split": args.split,
                "image": str(image_path.relative_to(output)), "question": row[args.question_key],
                "answers": answers, "answer_type": answer_type,
            }
        if not seen:
            raise ValueError("No examples matched the requested split and language")

    path = output / (args.split + ".jsonl")
    write_jsonl(path, convert())
    return {"file": str(path), "queries": len(seen)}
