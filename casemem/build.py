from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from casemem.data import batches, open_image, path_digests, png_bytes
from casemem.memory import Bank, Instance
from casemem.prompts import Request, question_prompt
from casemem.retrieval import normalized_box, pixel_box
from casemem.scoring import parse_answer, score_answer


def _request(example, original, crop=None):
    images = (original,) if crop is None else (original, crop)
    return Request(question_prompt(example.question, crop=crop is not None), images)


def _correct(example, completion, scoring):
    score = score_answer(parse_answer(completion.text, allow_bare=True), example.answers,
                         example.answer_type, scoring.open_metric)
    return score >= (scoring.open_correct_threshold if example.answer_type == "open" else 1.0)


def build_bank(examples, *, exclusions, encoder, teacher, detector, answerer, config, path, resume=False):
    if any(example.split not in {None, "train"} for example in examples):
        raise ValueError("Build the memory bank from training examples only")
    exclusion_keys = set(path_digests(exclusions).values())
    hashes = path_digests(examples)
    train = [example for example in examples if hashes[example.image] not in exclusion_keys]
    if not train:
        raise ValueError("No training examples remain after image exclusion")
    signature = {
        "train": [(example.row(), hashes[example.image]) for example in train],
        "models": asdict(config.models), "memory": asdict(config.memory),
        "scoring": asdict(config.scoring), "inference": asdict(config.inference),
    }
    build_id = sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    output = Path(path)
    if resume:
        bank = Bank.load(output)
        if bank.build_id != build_id:
            raise ValueError("Cannot resume: training images, answerer or build configuration changed")
    else:
        if output.exists():
            raise FileExistsError(f"{output} exists; use --resume or another output path")
        bank = Bank(config.models.text_encoder, config.models.visual_encoder, build_id=build_id)
        bank.stats.update(training_queries=len(train), excluded_queries=len(examples) - len(train),
                          failed_answers=0, proposed_regions=0, verified_regions=0)
        bank.save(output)

    remaining = [example for example in train if example.id not in bank.completed]
    for batch in batches(remaining, config.inference.batch_size):
        images = [open_image(example.image) for example in batch]
        initial = answerer.generate([_request(example, image) for example, image in zip(batch, images, strict=True)])
        for example, image, initial_answer in zip(batch, images, initial, strict=True):
            if not _correct(example, initial_answer, config.scoring):
                bank.stats["failed_answers"] += 1
                proposals = teacher.propose(image, example.question, config.memory.max_teacher_regions)
                bank.stats["proposed_regions"] += len(proposals)
                regions = []
                for proposal in proposals:
                    box = detector.refine(image, proposal)
                    if box is not None:
                        absolute = pixel_box(box, image.width, image.height)
                        if absolute[2] > absolute[0] and absolute[3] > absolute[1]:
                            regions.append((proposal, image.crop(absolute), normalized_box(absolute, image.width, image.height)))
                if regions:
                    repeated = answerer.generate([_request(example, image, crop) for _, crop, _ in regions])
                    kept = [region for region, response in zip(regions, repeated, strict=True)
                            if _correct(example, response, config.scoring)]
                    if kept:
                        features = encoder.images([image, *[crop for _, crop, _ in kept]])
                        full_feature = np.asarray(features[0], dtype=np.float32)
                        for (proposal, crop, box), vector in zip(kept, features[1:], strict=True):
                            item = Instance(example.id, hashes[example.image], box, proposal.concept, full_feature,
                                            np.asarray(vector, dtype=np.float32), png_bytes(crop))
                            bank.add(proposal.description, item, encoder, config.memory.merge_threshold,
                                     config.memory.guide_descriptions)
                            bank.stats["verified_regions"] += 1
            bank.completed.add(example.id)
        bank.save(output)
        print(f"Memory: {len(bank.completed)}/{len(train)} queries, {bank.stats['verified_regions']} verified regions", flush=True)
    return bank
