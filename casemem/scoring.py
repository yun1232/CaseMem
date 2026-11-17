from collections import Counter
import json
import math
import re
import unicodedata


ANSWER = re.compile(r"<answer>([^<>]*?)</answer>", re.IGNORECASE | re.DOTALL)
BOX = re.compile(r"<bbox>([^<>]*?)</bbox>", re.IGNORECASE | re.DOTALL)


def parse_answer(text: str, *, allow_bare: bool = False) -> str:
    matches = ANSWER.findall(text)
    if matches:
        return matches[-1].strip()
    if allow_bare and "<" not in text and "\n" not in text.strip():
        return text.strip()
    return ""


def normalize_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().replace("−", "-")
    return " ".join(re.findall(r"(?<!\w)[+-]?\d+(?:[.,]\d+)*(?!\w)|[^\W_]+", text, re.UNICODE))


def score_answer(prediction: str, references, answer_type: str, open_metric: str) -> float:
    prediction = normalize_answer(prediction)
    if not prediction:
        return 0.0
    references = (references,) if isinstance(references, str) else references
    predictions = Counter(prediction.split())
    scores = []
    for reference in references:
        expected = normalize_answer(reference)
        if answer_type in {"closed", "choice"} or open_metric == "exact":
            scores.append(float(prediction == expected))
        else:
            expected_tokens = Counter(expected.split())
            overlap = sum((predictions & expected_tokens).values())
            scores.append(2 * overlap / (sum(predictions.values()) + sum(expected_tokens.values())))
    return max(scores, default=0.0)


def valid_box(values, *, limit: float = 1.0):
    if values is None or isinstance(values, (str, bytes, dict)):
        return None
    try:
        items = tuple(values)
    except TypeError:
        return None
    if len(items) != 4 or any(isinstance(item, (bool, str)) for item in items):
        return None
    try:
        box = tuple(float(item) for item in items)
    except (ValueError, TypeError, OverflowError):
        return None
    if any(not math.isfinite(item) or not 0 <= item <= limit for item in box):
        return None
    return box if box[2] > box[0] and box[3] > box[1] else None


def parse_box(text: str):
    matches = BOX.findall(text)
    if len(matches) != 1:
        return None
    try:
        values = json.loads(matches[0])
    except json.JSONDecodeError:
        return None
    box = valid_box(values, limit=1000.0)
    return tuple(value / 1000.0 for value in box) if box else None


def box_iou(left, right) -> float:
    if left is None or right is None:
        return 0.0
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    overlap = width * height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - overlap
    return overlap / union if union > 0 else 0.0
