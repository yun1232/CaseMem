from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class Request:
    text: str
    images: tuple[Image.Image, ...]


def question_prompt(question: str, *, crop: bool = False, predict_box: bool = False) -> str:
    # The training parquet reserves these tokens for actual image placeholders.
    question = question.replace("<image>", "[image]").replace("<video>", "[video]")
    parts = [f"Question: {question}"]
    if crop:
        parts.append("The first image is the original. The second is a candidate crop from it. Interpret the crop in the context of the original image.")
    if predict_box:
        parts.append(
            "Locate the evidence in the original image. Return <bbox>[xmin, ymin, xmax, ymax]</bbox> "
            "with coordinates in [0,1000] in the original image, or <bbox>null</bbox> if no region applies. "
            "Then give a brief explanation in <think>...</think> and the final answer in <answer>...</answer>."
        )
    else:
        parts.append("Give only the concise final answer in <answer>...</answer>.")
    return "\n\n".join(parts)
