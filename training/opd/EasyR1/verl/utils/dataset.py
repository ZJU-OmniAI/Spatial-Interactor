# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    video_fps: float,
    video_nframes: Optional[int] = None,
    video_max_frames: Optional[int] = None,
    return_fps: bool = False,
    return_metadata: bool = False,
) -> Any:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels}
    if video_nframes is not None:
        vision_info["nframes"] = int(video_nframes)
    else:
        vision_info["fps"] = video_fps
        if video_max_frames is not None:
            vision_info["max_frames"] = int(video_max_frames)
    return fetch_video(vision_info, return_video_sample_fps=return_fps, return_video_metadata=return_metadata)


def build_position_ids(
    processor: Optional[ProcessorMixin],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    model_inputs: dict[str, Any],
) -> torch.Tensor:
    if processor is not None and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        if "Qwen3VLProcessor" in processor.__class__.__name__:
            from ..models.transformers.qwen3_vl import get_rope_index
        else:
            from ..models.transformers.qwen2_vl import get_rope_index

        vision_position_ids = get_rope_index(
            processor,
            input_ids=input_ids,
            image_grid_thw=model_inputs.get("image_grid_thw", None),
            video_grid_thw=model_inputs.get("video_grid_thw", None),
            second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
            attention_mask=attention_mask,
        )
        text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)
        return torch.cat((text_position_ids, vision_position_ids), dim=0)

    return torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        privileged_prompt_key: Optional[str] = None,
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        video_nframes: Optional[int] = None,
        video_max_frames: Optional[int] = None,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        assistant_prefill: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.is_qwen3_vl = processor is not None and "Qwen3VLProcessor" in processor.__class__.__name__
        self.prompt_key = prompt_key
        self.privileged_prompt_key = privileged_prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.video_nframes = video_nframes
        self.video_max_frames = video_max_frames
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.system_prompt = str(system_prompt or "").strip()
        self.assistant_prefill = str(assistant_prefill or "")

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _build_messages(
        self, example: dict[str, Any], prompt_key: Optional[str] = None
    ) -> list[dict[str, Any]]:
        prompt_str: str = example[prompt_key or self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            user_content = content_list
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            user_content = content_list
        else:
            user_content = prompt_str

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _render_chat(self, messages: list[dict[str, Any]]) -> str:
        renderer = self.processor if self.processor is not None else self.tokenizer
        prompt = renderer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return prompt + self.assistant_prefill

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self._render_chat(messages)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self._render_chat(messages)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_metadata = []
            for video in videos:
                processed_video = process_video(
                    video,
                    self.min_pixels,
                    self.max_pixels,
                    self.video_fps,
                    self.video_nframes,
                    self.video_max_frames,
                    return_metadata=self.is_qwen3_vl,
                )
                if self.is_qwen3_vl:
                    processed_video, metadata = processed_video
                    video_metadata.append(metadata)
                processed_videos.append(processed_video)

            model_inputs = self.processor(
                videos=processed_videos,
                text=[prompt],
                add_special_tokens=False,
                return_tensors="pt",
                **(
                    {"do_sample_frames": False, "video_metadata": video_metadata}
                    if self.is_qwen3_vl
                    else {}
                ),
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.encode(self._render_chat(messages), add_special_tokens=False)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        example["audit_prompt"] = str(example.get(self.prompt_key) or "")
        messages = self._build_messages(example)
        privileged_messages = None
        if self.privileged_prompt_key is not None:
            privileged_value = str(example.get(self.privileged_prompt_key) or "").strip()
            if not privileged_value:
                raise RuntimeError(
                    f"Missing non-empty {self.privileged_prompt_key!r} for dataset row {index}."
                )
            example["audit_privileged_prompt"] = privileged_value
            privileged_messages = self._build_messages(example, self.privileged_prompt_key)
        example.pop(self.prompt_key, None)
        if self.privileged_prompt_key is not None:
            example.pop(self.privileged_prompt_key, None)

        if self.image_key in example:
            prompt = self._render_chat(messages)
            privileged_prompt = (
                self._render_chat(privileged_messages)
                if privileged_messages is not None
                else None
            )
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]
            example["audit_media_path"] = str(images[0]) if images else ""

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            privileged_model_inputs = (
                self.processor(
                    processed_images,
                    [privileged_prompt],
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                if privileged_prompt is not None
                else None
            )
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif self.video_key in example:
            prompt = self._render_chat(messages)
            privileged_prompt = (
                self._render_chat(privileged_messages)
                if privileged_messages is not None
                else None
            )
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]
            example["audit_media_path"] = str(videos[0]) if videos else ""

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            video_metadata = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video,
                    self.min_pixels,
                    self.max_pixels,
                    self.video_fps,
                    self.video_nframes,
                    self.video_max_frames,
                    return_fps=True,
                    return_metadata=self.is_qwen3_vl,
                )
                if self.is_qwen3_vl:
                    processed_video, metadata = processed_video
                    video_metadata.append(metadata)
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos,
                text=[prompt],
                add_special_tokens=False,
                return_tensors="pt",
                **(
                    {"do_sample_frames": False, "video_metadata": video_metadata}
                    if self.is_qwen3_vl
                    else {}
                ),
            )
            privileged_model_inputs = (
                self.processor(
                    videos=processed_videos,
                    text=[privileged_prompt],
                    add_special_tokens=False,
                    return_tensors="pt",
                    **(
                        {"do_sample_frames": False, "video_metadata": video_metadata}
                        if self.is_qwen3_vl
                        else {}
                    ),
                )
                if privileged_prompt is not None
                else None
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]
                if privileged_model_inputs is not None:
                    privileged_model_inputs["second_per_grid_ts"] = [
                        2.0 / video_sample_fps for video_sample_fps in video_fps_list
                    ]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            example["audit_media_path"] = ""
            prompt = self._render_chat(messages)
            privileged_prompt = (
                self._render_chat(privileged_messages)
                if privileged_messages is not None
                else None
            )
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            privileged_model_inputs = (
                self.tokenizer(
                    [privileged_prompt], add_special_tokens=False, return_tensors="pt"
                )
                if privileged_prompt is not None
                else None
            )
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        position_ids = build_position_ids(self.processor, input_ids, attention_mask, model_inputs)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        if privileged_model_inputs is not None:
            privileged_input_ids = privileged_model_inputs.pop("input_ids")[0]
            privileged_attention_mask = privileged_model_inputs.pop("attention_mask")[0]
            privileged_position_ids = build_position_ids(
                self.processor,
                privileged_input_ids,
                privileged_attention_mask,
                privileged_model_inputs,
            )
            privileged_input_ids, privileged_attention_mask, privileged_position_ids = VF.postprocess_data(
                input_ids=privileged_input_ids,
                attention_mask=privileged_attention_mask,
                position_ids=privileged_position_ids,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        if privileged_model_inputs is not None:
            example["privileged_input_ids"] = privileged_input_ids
            example["privileged_attention_mask"] = privileged_attention_mask
            example["privileged_position_ids"] = privileged_position_ids
        example["ground_truth"] = example.pop(self.answer_key)
        return example
