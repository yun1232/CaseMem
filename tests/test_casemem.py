from collections import deque
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from casemem.backends import Completion, Proposal, parse_proposals
from casemem.build import build_bank
from casemem.config import Config, Memory, Retrieval
from casemem.data import Example, image_digest, prepare_splits
from casemem.memory import Bank, Instance
from casemem.pipeline import evaluate, save_training_parquet
from casemem.retrieval import Retriever
from casemem.scoring import box_iou, parse_box
from casemem.verl_reward import compute_score


class DummyEncoder:
    def text(self, texts):
        return np.asarray([[1.0, 0.0] if "lung" in text.lower() else [0.0, 1.0] for text in texts])

    def images(self, images):
        # Non-unit features exercise normalization after prototype aggregation.
        return np.asarray([[3.0, 0.0] for _ in images])


class QueueAnswerer:
    def __init__(self, answers):
        self.answers = deque(answers)
        self.image_counts = []

    def generate(self, requests):
        self.image_counts.extend(len(request.images) for request in requests)
        return [Completion(f"<answer>{self.answers.popleft()}</answer>", 7) for _ in requests]


class Teacher:
    def __init__(self):
        self.calls = 0

    def propose(self, image, question, count):
        self.calls += 1
        return [
            Proposal("lung", "center", "visible finding", (0.4, 0.4, 0.6, 0.6)),
            Proposal("lung", "center", "other finding", (0.4, 0.4, 0.6, 0.6)),
        ]


class Detector:
    def refine(self, image, proposal):
        return proposal.coarse_box


class CaseMemTest(unittest.TestCase):
    def setUp(self):
        work = TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.root = Path(work.name)
        self.config = Config()

    def sample(self, identifier, color, split=None):
        path = self.root / f"{identifier}.png"
        Image.new("RGB", (96, 64), color).save(path)
        return Example(identifier, "VQA-RAD", path, "Is the lung visible?", ("yes",), "closed", split)

    def bank(self, *instances):
        bank = Bank(self.config.models.text_encoder, self.config.models.visual_encoder)
        for entry in instances:
            bank.add("lung", entry, DummyEncoder(), 0.9, 4)
        return bank

    def memory(self, box, source="source"):
        return Instance(source, source, box, "lung", np.array([2.0, 0.0]), np.array([4.0, 0.0]), b"test")

    def test_dedup_by_pixels_across_different_filenames(self):
        train = [self.sample("train-a", (25, 0, 0)), self.sample("train-b", (85, 0, 0)),
                 self.sample("train-c", (120, 0, 0))]
        val = [self.sample("val-a", (85, 0, 0))]
        test = [self.sample("test-a", (25, 0, 0))]
        filtered, validation = prepare_splits(train, val, test)
        self.assertEqual([row.id for row in filtered], ["train-c"])
        self.assertEqual(validation[0].split, "validation")

    def test_frozen_bank_roundtrip_and_no_self_match(self):
        path = self.root / "memory.zip"
        bank = self.bank(self.memory((0.3, 0.3, 0.7, 0.7)))
        bank.completed.add("train-1")
        bank.save(path)
        reloaded = Bank.load(path)
        self.assertEqual(reloaded.completed, {"train-1"})
        self.assertEqual(reloaded.clusters[0].instances[0].crop_png, b"test")
        image = Image.new("RGB", (80, 60))
        selector = Retriever(reloaded, DummyEncoder(), self.config.retrieval)
        self.assertFalse(selector.select(image, "lung", source_hash="source").active)
        self.assertTrue(selector.select(image, "lung").active)

    def test_spatial_gate_and_normalized_prototype(self):
        selector = Retriever(self.bank(self.memory((0.3, 0.3, 0.7, 0.7))), DummyEncoder(), Retrieval())
        chosen = selector.select(Image.new("RGB", (96, 64)), "lung")
        self.assertTrue(chosen.active)
        self.assertAlmostEqual(chosen.visual, 1.0)
        self.assertAlmostEqual(chosen.spatial, 0.881, places=3)
        self.assertEqual(chosen.focus_box, (0.375, 0.375, 0.625, 0.625))
        distributed = self.bank(self.memory((0, 0, 0.2, 0.2), "one"), self.memory((0.8, 0.8, 1, 1), "two"))
        rejected = Retriever(distributed, DummyEncoder(), Retrieval()).select(Image.new("RGB", (96, 64)), "lung")
        self.assertGreater(rejected.score, 1.2)
        self.assertFalse(rejected.active)

    def test_build_verifies_failures_and_resume_is_idempotent(self):
        examples = [self.sample("yes", (50, 0, 0), "train"), self.sample("no", (80, 0, 0), "train")]
        answerer = QueueAnswerer(["yes", "no", "yes", "no"])
        teacher = Teacher()
        kwargs = {"exclusions": [], "encoder": DummyEncoder(), "teacher": teacher, "detector": Detector(),
                  "answerer": answerer, "config": self.config, "path": self.root / "bank.zip"}
        bank = build_bank(examples, **kwargs)
        self.assertEqual(teacher.calls, 1)
        self.assertEqual(answerer.image_counts, [1, 1, 2, 2])
        self.assertEqual(bank.stats["verified_regions"], 1)
        self.assertEqual(bank.clusters[0].instances[0].source_id, "no")
        resumed = build_bank(examples, resume=True, **kwargs)
        self.assertEqual(resumed.stats, bank.stats)
        self.assertEqual(teacher.calls, 1)
        changed = replace(self.config, memory=Memory(merge_threshold=0.8))
        with self.assertRaisesRegex(ValueError, "Cannot resume"):
            build_bank(examples, resume=True, **dict(kwargs, config=changed))

    def test_score_keeps_answer_reward_if_box_missing_and_supports_control(self):
        extra = {"memory_active": True, "focus_box": (0.0, 0.0, 0.5, 0.5), "answer_type": "closed"}
        exact = "<bbox>[0,0,500,500]</bbox><answer>yes</answer>"
        wrong_box = "<bbox>[300,0,100,500]</bbox><answer>yes</answer>"
        self.assertEqual(compute_score("vqarad", exact, '["yes"]', extra)["score"], 1.2)
        self.assertEqual(compute_score("vqarad", wrong_box, '["yes"]', extra)["score"], 1.0)
        self.assertEqual(compute_score("vqarad", exact, '["yes"]', extra, localization_weight=0)["score"], 1.0)
        self.assertEqual(compute_score("vqarad", exact, '["yes"]', dict(extra, memory_active=False))["score"], 1.0)

    def test_box_parser_is_strict_and_iou_matches_coordinates(self):
        parsed = parse_box("<bbox>[0,0,1000,500]</bbox>")
        self.assertEqual(box_iou(parsed, (0, 0, 1, 1)), 0.5)
        self.assertIsNone(parse_box("<bbox>[0,0,1001,500]</bbox>"))
        self.assertIsNone(parse_box("<bbox>[false,0,1000,500]</bbox>"))
        self.assertIsNone(parse_box("<bbox>[0,0,NaN,500]</bbox>"))
        self.assertIsNone(parse_box("<bbox>[0,0,1000,500]</bbox><bbox>[0,0,1000,500]</bbox>"))

    def test_training_parquet_has_only_one_image_and_reward_metadata(self):
        import pyarrow.parquet as pq
        example = self.sample("training", (100, 0, 0), "train")
        retriever = Retriever(self.bank(self.memory((0.3, 0.3, 0.7, 0.7))), DummyEncoder(), Retrieval())
        target = self.root / "train.parquet"
        summary = save_training_parquet([example], retriever, self.config.scoring, target)
        self.assertEqual(summary["localized"], 1)
        row = pq.read_table(target).to_pylist()[0]
        self.assertEqual(len(row["images"]), 1)
        self.assertTrue(row["extra_info"]["memory_active"])
        self.assertNotIn("0.375", row["prompt"][0]["content"])
        self.assertEqual(row["extra_info"]["focus_box"], [0.375, 0.375, 0.625, 0.625])
        self.assertEqual(row["prompt"][0]["content"].count("<image>"), 1)

    def test_inference_crop_ablation_and_split_metrics(self):
        retriever = Retriever(self.bank(self.memory((0.3, 0.3, 0.7, 0.7))), DummyEncoder(), Retrieval())
        examples = [self.sample("query", (75, 0, 0), "test")]
        with_crop, no_crop = QueueAnswerer(["yes"]), QueueAnswerer(["no"])
        results, metrics = evaluate(examples, with_crop, self.config, retriever=retriever)
        control, no_crop_metrics = evaluate(examples, no_crop, self.config, retriever=retriever, crop=False)
        self.assertTrue(results[0]["crop_applied"])
        self.assertFalse(control[0]["crop_applied"])
        self.assertEqual(metrics["success"], {"N": 1, "C": 1, "accuracy": 1.0})
        self.assertEqual(no_crop_metrics["success"], {"N": 1, "C": 0, "accuracy": 0.0})
        self.assertEqual(with_crop.image_counts, [2])
        self.assertEqual(no_crop.image_counts, [1])

    def test_evaluation_rejects_test_images_in_bank(self):
        sample = self.sample("query", (75, 0, 0), "test")
        with Image.open(sample.image) as image:
            digest = image_digest(image)
        bank = self.bank(self.memory((0.3, 0.3, 0.7, 0.7), digest))
        retriever = Retriever(bank, DummyEncoder(), Retrieval())
        with self.assertRaisesRegex(ValueError, "also occur in the memory bank"):
            evaluate([sample], QueueAnswerer([]), self.config, retriever=retriever)

    def test_teacher_parses_optional_normalized_coordinates(self):
        proposals = parse_proposals(
            '```json\n{"regions":[{"concept":"lung","location":"right","attributes":"opacity","bbox_2d":[0,0,500,500]}]}\n```', 3
        )
        self.assertEqual(proposals[0].coarse_box, (0.0, 0.0, 0.5, 0.5))
        self.assertEqual(proposals[0].description, "lung. right. opacity")


if __name__ == "__main__":
    unittest.main()
