"""CaseMem reward function for VeRL."""

import json

from casemem.scoring import box_iou, parse_answer, parse_box, score_answer, valid_box


def compute_score(data_source, solution_str, ground_truth, extra_info=None, localization_weight=0.2):
    extra = extra_info or {}
    # Parquet stores the answer alternatives as a JSON string.
    references = (ground_truth,)
    if isinstance(ground_truth, str):
        try:
            decoded = json.loads(ground_truth)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list) and all(isinstance(item, str) for item in decoded):
            references = decoded
    elif isinstance(ground_truth, (list, tuple)):
        references = ground_truth
    task = score_answer(parse_answer(solution_str), references, extra.get("answer_type", "open"), extra.get("open_metric", "exact"))
    predicted = parse_box(solution_str)
    focus = valid_box(extra.get("focus_box")) if extra.get("memory_active", False) else None
    loc = box_iou(predicted, focus)
    correct = float(task >= float(extra.get("correct_threshold", 1.0)))
    return {
        "score": task + localization_weight * loc,
        "acc": correct,
        "task_reward": task,
        "localization_reward": loc,
        "memory_active": float(focus is not None),
        "valid_prediction_box": float(predicted is not None),
    }
