import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def training_command(args, config):
    settings = config.training
    world_size = args.gpus * args.nodes
    if min(args.gpus, args.nodes, args.tensor_parallel) < 1 or args.gpus % args.tensor_parallel:
        raise ValueError("Each node's GPU count must be divisible by tensor parallel size")
    if (settings.queries_per_batch * settings.rollouts_per_query) % world_size:
        raise ValueError("The rollout count in a batch must be divisible by the total number of training GPUs")

    root = Path(__file__).resolve().parent.parent
    model = args.model or config.models.backbone
    weight = 0.0 if args.answer_only else settings.localization_weight
    memory = args.gpu_memory_utilization
    if not 0 < memory < 1:
        raise ValueError("Rollout GPU memory utilization must be in (0, 1)")

    def quote(path):
        return json.dumps(str(path))

    train, val, output = (Path(value).resolve() for value in (args.train, args.val, args.output))
    if train == val:
        raise ValueError("Use separate training and validation Parquet files")
    reward = "reward.custom_reward_function"
    command = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        f"data.train_files=[{quote(train)}]", f"data.val_files=[{quote(val)}]",
        f"data.train_batch_size={settings.queries_per_batch}", "data.val_batch_size=64",
        f"data.max_prompt_length={settings.prompt_tokens}", f"data.max_response_length={settings.response_tokens}",
        "data.filter_overlong_prompts=True", "data.truncation=error",
        "algorithm.adv_estimator=grpo", "algorithm.norm_adv_by_std_in_grpo=True",
        "algorithm.use_kl_in_reward=False",
        f"actor_rollout_ref.model.path={quote(model)}", "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        f"actor_rollout_ref.actor.optim.lr={settings.learning_rate}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={settings.queries_per_batch}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.rollout.name=vllm", f"actor_rollout_ref.rollout.n={settings.rollouts_per_query}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.tensor_parallel}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={memory}",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.temperature=1.0", "actor_rollout_ref.rollout.mode=async",
        "reward.reward_manager.name=naive",
        f"actor_rollout_ref.rollout.max_model_len={settings.prompt_tokens + settings.response_tokens}",
        f"{reward}.path={quote(Path(__file__).with_name('verl_reward.py').resolve())}",
        f"{reward}.name=compute_score", f"+{reward}.reward_kwargs.localization_weight={weight}",
        f"trainer.n_gpus_per_node={args.gpus}", f"trainer.nnodes={args.nodes}",
        f"trainer.total_training_steps={settings.steps}", f"trainer.total_epochs={settings.steps}",
        f"trainer.default_local_dir={quote(output)}", "trainer.logger=[console]",
        "trainer.project_name=CaseMem",
        f"trainer.experiment_name={quote('answer-only' if args.answer_only else 'localization-grpo')}",
        f"trainer.resume_mode={'auto' if args.resume else 'disable'}",
        "trainer.save_freq=50", "trainer.test_freq=25", "trainer.val_before_train=False",
        "actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model,optimizer,extra]",
    ]
    env = os.environ.copy()
    python_paths = [str(root)]
    if args.verl_source:
        upstream = Path(args.verl_source).resolve()
        if not (upstream / "verl" / "trainer" / "main_ppo.py").is_file():
            raise FileNotFoundError(f"VeRL trainer not found under {upstream}")
        python_paths.insert(0, str(upstream))
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["PYTHONUNBUFFERED"] = "1"
    return command, env


def launch_training(args, config):
    command, env = training_command(args, config)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        for value in (args.train, args.val):
            if not Path(value).is_file():
                raise FileNotFoundError(f"Training data file not found: {value}")
        if not args.resume and (Path(args.output) / "latest_checkpointed_iteration.txt").exists():
            raise FileExistsError(f"{args.output} already contains checkpoints. Use --resume or another directory.")
        subprocess.run(command, env=env, check=True)
