import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

from casemem.backends import policy_image
from casemem.data import batches, open_image, path_digests, png_bytes
from casemem.prompts import Request, question_prompt
from casemem.retrieval import Selection
from casemem.scoring import parse_answer, parse_box, score_answer


def training_schema(pa):
    return pa.schema([
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("images", pa.list_(pa.struct([("bytes", pa.binary())]))),
        ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
        ("extra_info", pa.struct([
            ("index", pa.int64()), ("question_id", pa.string()), ("answer_type", pa.string()),
            ("open_metric", pa.string()), ("correct_threshold", pa.float64()), ("memory_active", pa.bool_()),
            ("focus_box", pa.list_(pa.float64())),
        ])),
    ])


def _check_bank_separation(bank, hashes):
    stored = {item.image_hash for cl in bank.clusters for item in cl.instances}
    overlap = stored.intersection(hashes)
    if overlap:
        raise ValueError(f"{len(overlap)} evaluation images also occur in the memory bank")


def make_training_row(example, image, selection, scoring, index):
    return {
        "data_source": example.dataset,
        "prompt": [{"role": "user", "content": "<image>\n" + question_prompt(example.question, predict_box=True)}],
        "images": [{"bytes": png_bytes(policy_image(image))}],
        "reward_model": {"style": "rule", "ground_truth": json.dumps(example.answers, ensure_ascii=False)},
        "extra_info": {
            "index": index, "question_id": example.id, "answer_type": example.answer_type,
            "open_metric": scoring.open_metric,
            "correct_threshold": scoring.open_correct_threshold if example.answer_type == "open" else 1.0,
            "memory_active": selection.active, "focus_box": selection.focus_box if selection.active else None,
        },
    }


def save_training_parquet(examples, retriever, scoring, target, *, validation=False):
    import pyarrow as pa
    import pyarrow.parquet as pq

    hashes = path_digests(examples)
    if validation:
        _check_bank_separation(retriever.bank, set(hashes.values()))
    schema = training_schema(pa)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, delete=False) as file:
        temp = Path(file.name)
    active_count = 0
    try:
        with pq.ParquetWriter(temp, schema, compression="zstd") as writer:
            index = 0
            for batch in batches(examples, 32):
                rows = []
                for example in batch:
                    image = open_image(example.image)
                    selection = retriever.select(image, example.question, source_hash=hashes[example.image])
                    active_count += selection.active
                    rows.append(make_training_row(example, image, selection, scoring, index))
                    index += 1
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return {"queries": len(examples), "localized": active_count, "file": str(target)}


def eval_requests(examples, images, *, retriever, strategy, crop, predict_box):
    requests = []
    metadata = []
    for example, image in zip(examples, images, strict=True):
        selection = (retriever.select(image, example.question, strategy=strategy) if retriever else Selection())
        augmented = bool(crop and selection.active)
        context = (image, image.crop(selection.pixel_box)) if augmented else (image,)
        requests.append(Request(question_prompt(example.question, crop=augmented, predict_box=predict_box), context))
        metadata.append((selection, augmented))
    return requests, metadata


def evaluate(examples, answerer, config, *, retriever=None, strategy="full", crop=True, predict_box=False, warmup=0):
    if retriever is not None:
        _check_bank_separation(retriever.bank, set(path_digests(examples).values()))
    if warmup:
        first = examples[:config.inference.batch_size]
        images = [open_image(example.image) for example in first]
        for _ in range(warmup):
            requests, _ = eval_requests(first, images, retriever=retriever, strategy=strategy,
                                        crop=crop, predict_box=predict_box)
            answerer.generate(requests)
    details = []
    total_seconds = 0.0
    for batch in batches(examples, config.inference.batch_size):
        images = [open_image(example.image) for example in batch]
        start = perf_counter()
        requests, metadata = eval_requests(batch, images, retriever=retriever, strategy=strategy,
                                           crop=crop, predict_box=predict_box)
        completions = answerer.generate(requests)
        total_seconds += perf_counter() - start
        for example, completion, (selection, augmented) in zip(batch, completions, metadata, strict=True):
            answer = parse_answer(completion.text, allow_bare=not predict_box)
            score = score_answer(answer, example.answers, example.answer_type, config.scoring.open_metric)
            threshold = config.scoring.open_correct_threshold if example.answer_type == "open" else 1.0
            details.append({
                "id": example.id, "dataset": example.dataset, "answer_type": example.answer_type,
                "prediction": answer, "references": example.answers, "completion": completion.text,
                "correct": score >= threshold, "task_score": score, "generated_tokens": completion.generated_tokens,
                "retrieved": selection.active, "crop_applied": augmented,
                "candidate_count": len(selection.hits), "matched_cases": [hit.instance.source_id for hit in selection.hits],
                "focus_box": selection.focus_box if selection.active else None,
                "candidate_score": selection.score, "spatial_support": selection.spatial,
                "visual_similarity": selection.visual, "predicted_box": parse_box(completion.text),
            })
    groups = {
        "all": details,
        "success": [row for row in details if row["retrieved"]],
        "fail": [row for row in details if not row["retrieved"]],
        "open": [row for row in details if row["answer_type"] == "open"],
        "closed": [row for row in details if row["answer_type"] == "closed"],
        "choice": [row for row in details if row["answer_type"] == "choice"],
    }
    metrics = {key: correctness(rows) for key, rows in groups.items()}
    metrics["retrieval_rate"] = len(groups["success"]) / len(details)
    metrics["crop_coverage"] = sum(row["crop_applied"] for row in details) / len(details)
    metrics["output_tokens_per_query"] = sum(row["generated_tokens"] for row in details) / len(details)
    metrics["amortized_online_seconds_per_query"] = total_seconds / len(details)
    metrics["online_seconds_total"] = total_seconds
    return details, metrics


def correctness(rows):
    count = len(rows)
    correct = sum(row["correct"] for row in rows)
    return {"N": count, "C": correct, "accuracy": correct / count if count else None}
