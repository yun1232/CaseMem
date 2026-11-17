import argparse
import json
from pathlib import Path

from casemem.config import load_config
from casemem.data import load_examples, path_digests, prepare_splits, write_jsonl


def _data_args(parser, *, device=True):
    if device:
        parser.add_argument("--encoder-device", default="cuda")
        parser.add_argument("--encoder-batch-size", type=int, default=32)


def _vllm_args(parser):
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)


def build_parser():
    parser = argparse.ArgumentParser(description="CaseMem visual memory for MedVQA")
    parser.add_argument("--config", help="CaseMem JSON configuration, or omit to use the shipped defaults")
    commands = parser.add_subparsers(dest="command", required=True)

    importer = commands.add_parser("import-hf", help="Import an image QA split from Hugging Face")
    data_source = importer.add_mutually_exclusive_group(required=True)
    data_source.add_argument("--dataset-id")
    data_source.add_argument("--parquet", nargs="+", help="Use local Hugging Face-format Parquet shards")
    importer.add_argument("--dataset-name", required=True)
    importer.add_argument("--hf-split", required=True)
    importer.add_argument("--split", required=True, choices=("train", "validation", "test"))
    importer.add_argument("--subset")
    importer.add_argument("--output-dir", required=True)
    importer.add_argument("--image-key", default="image")
    importer.add_argument("--question-key", default="question")
    importer.add_argument("--answer-key", default="answer")
    importer.add_argument("--id-key")
    importer.add_argument("--type-key")
    importer.add_argument("--language")
    importer.add_argument("--language-key", default="q_lang")

    split = commands.add_parser("prepare-splits", help="Exclude evaluation images from training and write audited manifests")
    split.add_argument("--train", required=True)
    split.add_argument("--val", help="Optional validation manifest")
    split.add_argument("--validation-fraction", type=float, default=0.1)
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--test", required=True, nargs="+")
    split.add_argument("--output-dir", required=True)

    build = commands.add_parser("build-memory", help="Build the fixed memory bank from verified training failures")
    build.add_argument("--train", required=True)
    build.add_argument("--exclude", nargs="+", default=[])
    build.add_argument("--output", required=True)
    build.add_argument("--detector-device", default="cuda")
    build.add_argument("--resume", action="store_true")
    _data_args(build)
    _vllm_args(build)

    stage = commands.add_parser("prepare-grpo", help="Generate no-crop Parquet data with per-query localization metadata")
    stage.add_argument("--train", required=True)
    stage.add_argument("--val", required=True)
    stage.add_argument("--bank", required=True)
    stage.add_argument("--output-dir", required=True)
    _data_args(stage)

    train = commands.add_parser("train", help="Run CaseMem single-turn GRPO")
    train.add_argument("--train", required=True)
    train.add_argument("--val", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--model", help="Policy checkpoint or Hugging Face identifier")
    train.add_argument("--verl-source", help="Local checkout of VeRL v0.7.1, if not installed")
    train.add_argument("--gpus", type=int, default=4, help="Training GPUs per node")
    train.add_argument("--nodes", type=int, default=1)
    train.add_argument("--answer-only", action="store_true")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--dry-run", action="store_true")
    _vllm_args(train)

    infer = commands.add_parser("evaluate", help="Run direct or memory-guided inference and save sample-level metrics")
    infer.add_argument("--data", required=True)
    infer.add_argument("--bank", help="Optional fixed memory bank")
    infer.add_argument("--model", help="Policy checkpoint or Hugging Face identifier")
    infer.add_argument("--strategy", choices=("full", "spatial", "cosine"), default="full")
    infer.add_argument("--disable-crop", action="store_true")
    infer.add_argument("--predict-box", action="store_true", help="Use the localization-trained response format")
    infer.add_argument("--warmup", type=int, default=0, help="Exclude this many first-batch reruns from timing")
    infer.add_argument("--output-dir", required=True)
    _data_args(infer)
    _vllm_args(infer)

    inspect = commands.add_parser("inspect-bank", help="Report bank size and verification counts")
    inspect.add_argument("--bank", required=True)
    return parser


def report(metadata):
    print(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))


def _fail_if_not_split(examples, split):
    if any(example.split not in {None, split} for example in examples):
        raise ValueError(f"Expected only {split} examples")


def _encoders(args, config):
    from casemem.backends import Embeddings

    if args.encoder_batch_size < 1:
        raise ValueError("Encoder batch size must be positive")
    return Embeddings(config.models, args.encoder_device, args.encoder_batch_size)


def _answerer(args, config, model=None, max_images=2):
    from casemem.backends import VLLMAnswerer

    return VLLMAnswerer(model or config.models.backbone, config.inference, tensor_parallel=args.tensor_parallel,
                        gpu_memory_utilization=args.gpu_memory_utilization, max_images=max_images)


def _retriever(args, config, bank):
    from casemem.retrieval import Retriever

    bank.check_encoders(config.models)
    return Retriever(bank, _encoders(args, config) if bank.clusters else None, config.retrieval)


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args, load_config(args.config))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"CaseMem: {exc}\n")


def run(args, config):
    from casemem.memory import Bank

    if args.command == "import-hf":
        from casemem.import_data import import_huggingface

        report(import_huggingface(args))

    elif args.command == "prepare-splits":
        train = load_examples(args.train)
        validation = load_examples(args.val) if args.val else None
        tests = [example for path in args.test for example in load_examples(path)]
        filtered_train, filtered_val = prepare_splits(
            train, validation, tests, validation_fraction=args.validation_fraction, seed=args.seed,
        )
        output = Path(args.output_dir)
        write_jsonl(output / "train.jsonl", (example.row() for example in filtered_train))
        write_jsonl(output / "validation.jsonl", (example.row() for example in filtered_val))
        audit = {"train_input": len(train), "train_kept": len(filtered_train), "validation_input": len(validation) if validation is not None else None,
                 "validation_kept": len(filtered_val), "test_exclusion_queries": len(tests)}
        (output / "split_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        report(audit)

    elif args.command == "build-memory":
        from casemem.backends import GroundingDino, QwenFlashTeacher
        from casemem.build import build_bank

        train = load_examples(args.train)
        _fail_if_not_split(train, "train")
        exclude = [example for path in args.exclude for example in load_examples(path)]
        teacher = QwenFlashTeacher(config.models.teacher)
        bank = build_bank(train, exclusions=exclude, encoder=_encoders(args, config), teacher=teacher,
                          detector=GroundingDino(config.models.detector, config.memory, args.detector_device),
                          answerer=_answerer(args, config), config=config, path=args.output, resume=args.resume)
        report(bank.stats)

    elif args.command == "prepare-grpo":
        from casemem.pipeline import save_training_parquet

        train, val = load_examples(args.train), load_examples(args.val)
        _fail_if_not_split(train, "train")
        _fail_if_not_split(val, "validation")
        digests = path_digests([*train, *val])
        if {digests[row.image] for row in train} & {digests[row.image] for row in val}:
            raise ValueError("Training and validation images overlap. Run prepare-splits first.")
        retriever = _retriever(args, config, Bank.load(args.bank))
        output = Path(args.output_dir)
        report({"train": save_training_parquet(train, retriever, config.scoring, output / "train.parquet"),
                "validation": save_training_parquet(val, retriever, config.scoring, output / "validation.parquet",
                                                    validation=True)})

    elif args.command == "train":
        from casemem.train import launch_training

        launch_training(args, config)

    elif args.command == "evaluate":
        from casemem.pipeline import evaluate

        if args.warmup < 0:
            raise ValueError("Warmup count must be non-negative")
        examples = load_examples(args.data)
        if any(example.split == "train" for example in examples):
            raise ValueError("Use test or validation data for evaluation")
        bank = Bank.load(args.bank) if args.bank else None
        retriever = _retriever(args, config, bank) if bank is not None else None
        max_images = 2 if bank is not None and not args.disable_crop else 1
        details, metrics = evaluate(
            examples, _answerer(args, config, args.model, max_images=max_images), config,
            retriever=retriever, strategy=args.strategy, crop=not args.disable_crop,
            predict_box=args.predict_box, warmup=args.warmup,
        )
        metrics.update({"model": args.model or config.models.backbone, "strategy": args.strategy if bank else "direct",
                        "bank": args.bank, "batch_size": config.inference.batch_size, "warmup_batches": args.warmup,
                        "crop_enabled": bank is not None and not args.disable_crop, "predict_box": args.predict_box})
        output = Path(args.output_dir)
        write_jsonl(output / "predictions.jsonl", details)
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        report(metrics)

    else:
        bank = Bank.load(args.bank)
        report({"clusters": len(bank.clusters), "instances": sum(len(cl.instances) for cl in bank.clusters),
                "processed_queries": len(bank.completed), "text_encoder": bank.text_encoder,
                "visual_encoder": bank.visual_encoder, **bank.stats})
