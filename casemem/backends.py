from base64 import b64encode
from dataclasses import dataclass
from io import BytesIO
import json
import os
import re
from time import sleep

import numpy as np
from PIL import Image

from casemem.data import batches
from casemem.retrieval import normalized_box
from casemem.scoring import box_iou, valid_box


POLICY_IMAGE_MAX_SIDE = 768


@dataclass(frozen=True)
class Proposal:
    concept: str
    location: str
    attributes: str
    coarse_box: tuple[float, float, float, float] | None

    @property
    def description(self):
        return ". ".join(part for part in (self.concept, self.location, self.attributes) if part)


@dataclass(frozen=True)
class Completion:
    text: str
    generated_tokens: int


def policy_image(image: Image.Image, max_side: int = POLICY_IMAGE_MAX_SIDE) -> Image.Image:
    if max(image.size) <= max_side:
        return image
    resized = image.copy()
    resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return resized


class Embeddings:
    def __init__(self, models, device: str, batch_size: int = 32):
        import open_clip
        import torch
        from sentence_transformers import SentenceTransformer

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.text_model = SentenceTransformer(models.text_encoder, device=device).eval()
        self.vision_model, self.preprocess = open_clip.create_model_from_pretrained(
            f"hf-hub:{models.visual_encoder}", device=device
        )
        self.vision_model.eval()

    def text(self, texts):
        return self.text_model.encode(texts, batch_size=self.batch_size, show_progress_bar=False,
                                      convert_to_numpy=True, normalize_embeddings=False)

    def images(self, images):
        parts = []
        with self.torch.inference_mode():
            for chunk in batches(images, self.batch_size):
                pixels = self.torch.stack([self.preprocess(image) for image in chunk]).to(self.device)
                features = self.vision_model.encode_image(pixels, normalize=False)
                parts.append(features.float().cpu().numpy())
        return np.concatenate(parts, axis=0)


def parse_proposals(content: str, count: int):
    content = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", content, flags=re.IGNORECASE)
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Teacher response was not valid JSON") from exc
    if not isinstance(result, dict) or not isinstance(result.get("regions"), list):
        raise ValueError("Teacher response needs a regions list")
    proposals = []
    for row in result["regions"]:
        if not isinstance(row, dict) or not isinstance(row.get("concept"), str):
            continue
        concept = row["concept"].strip()
        if not concept:
            continue
        box = valid_box(row.get("bbox_2d"), limit=1000.0)
        proposals.append(Proposal(concept, str(row.get("location") or "").strip(),
                                  str(row.get("attributes") or "").strip(),
                                  tuple(value / 1000.0 for value in box) if box else None))
        if len(proposals) >= count:
            break
    return proposals


class QwenFlashTeacher:
    def __init__(self, model):
        from openai import OpenAI

        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("Set DASHSCOPE_API_KEY before building the memory bank")
        self.client = OpenAI(
            api_key=api_key, max_retries=3, timeout=90.0,
            base_url=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        )
        self.model = model

    def propose(self, image, question, count):
        with BytesIO() as buffer:
            policy_image(image, max_side=1600).save(buffer, "JPEG", quality=93)
            image_url = "data:image/jpeg;base64," + b64encode(buffer.getvalue()).decode("ascii")
        prompt = (
            f"Propose at most {count} visible regions that provide evidence for answering the question. "
            "Return a JSON object with a regions array. For each entry give a short grounding concept, "
            "its approximate location, visible attributes, and bbox_2d as [xmin,ymin,xmax,ymax] "
            "in [0,1000] relative to the original image. Use an empty list if nothing can be localized. "
            "Do not answer the question or include any text outside the JSON object.\n"
            f"Question: {question}"
        )
        for attempt in range(3):
            result = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ]}],
                temperature=0, max_tokens=768, extra_body={"enable_thinking": False},
            )
            try:
                return parse_proposals(result.choices[0].message.content or "", count)
            except ValueError:
                if attempt == 2:
                    raise
                sleep(1)


class GroundingDino:
    def __init__(self, model, memory, device):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model).to(device).eval()
        self.memory = memory

    def refine(self, image, proposal):
        label = proposal.concept.casefold().strip().rstrip(".") + "."
        inputs = self.processor(images=image, text=label, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        matches = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.memory.detector_box_threshold,
            text_threshold=self.memory.detector_text_threshold, target_sizes=[(image.height, image.width)]
        )[0]
        best_key = None
        best_box = None
        for coordinates, confidence in zip(matches["boxes"].tolist(), matches["scores"].tolist(), strict=True):
            scaled = normalized_box(coordinates, image.width, image.height)
            box = valid_box(tuple(min(1.0, max(0.0, point)) for point in scaled))
            if box is None:
                continue
            key = (box_iou(box, proposal.coarse_box), confidence)
            if best_key is None or key > best_key:
                best_key, best_box = key, box
        return best_box


class VLLMAnswerer:
    def __init__(self, model, inference, *, tensor_parallel=1, gpu_memory_utilization=0.6, max_images=2):
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self.processor = AutoProcessor.from_pretrained(model)
        self.llm = LLM(
            model=model, tensor_parallel_size=tensor_parallel, limit_mm_per_prompt={"image": max_images, "video": 0},
            mm_processor_kwargs={"size": {"shortest_edge": 56 ** 2, "longest_edge": POLICY_IMAGE_MAX_SIDE ** 2}},
            max_model_len=inference.max_sequence_tokens, gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=inference.batch_size, seed=0,
        )
        self.parameters = SamplingParams(temperature=0, max_tokens=inference.max_new_tokens)

    def generate(self, requests):
        inputs = []
        for request in requests:
            content = [{"type": "image"} for _ in request.images]
            content.append({"type": "text", "text": request.text})
            prompt = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
            )
            images = [policy_image(image) for image in request.images]
            inputs.append({"prompt": prompt, "multi_modal_data": {"image": images}})
        outputs = self.llm.generate(inputs, self.parameters, use_tqdm=False)
        return [Completion(item.outputs[0].text, len(item.outputs[0].token_ids)) for item in outputs]
