import json
import os
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import log_rank


class DistillDataset(Dataset):
    def __init__(
        self,
        args,
        split: str,
        student_tokenizer,
        teacher_configs: Optional[List[dict]] = None,
    ):
        self.args = args
        self.split = split
        self.student_tokenizer = student_tokenizer
        self.teacher_configs = teacher_configs or []
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.dataset = self._load_and_process_data()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def _load_and_process_data(self):
        dataset = []
        path = os.path.join(self.args.data_dir, f"{self.split}.jsonl")

        if not os.path.exists(path):
            raise FileNotFoundError(f"No such file named {path}")

        with open(path) as file_obj:
            raw_data = [json.loads(line) for line in file_obj.readlines()]
            self.answers = [
                item["output"] if isinstance(item["output"], list) else [item["output"]]
                for item in raw_data
            ]

        log_rank("Processing dataset for student model (and all teacher models)...")
        seg = np.iinfo(np.int32).max * 2 + 1
        for data in tqdm(raw_data, disable=(dist.get_rank() != 0)):
            student_prompt_ids = self.student_tokenizer.encode(
                data["prompt"], add_special_tokens=False
            )
            student_prompt_ids = student_prompt_ids[: self.max_prompt_length]
            student_response_ids = self.student_tokenizer.encode(
                data["output"], add_special_tokens=False
            )
            student_response_ids = student_response_ids + [
                self.student_tokenizer.eos_token_id
            ]

            tokenized_data = {
                "student_input_ids": student_prompt_ids + [seg] + student_response_ids,
            }

            for teacher in self.teacher_configs:
                teacher_tokenizer = teacher["tokenizer"]
                teacher_prompt_ids = teacher_tokenizer.encode(
                    data["prompt"], add_special_tokens=False
                )
                teacher_prompt_ids = teacher_prompt_ids[: self.max_prompt_length]
                teacher_response_ids = teacher_tokenizer.encode(
                    data["output"], add_special_tokens=False
                )
                teacher_response_ids = teacher_response_ids + [
                    teacher_tokenizer.eos_token_id
                ]
                tokenized_data[
                    f'teacher_{teacher["id"]}_input_ids'
                ] = teacher_prompt_ids + [seg] + teacher_response_ids

            dataset.append(tokenized_data)

        return dataset

    def _process_lm(
        self,
        index,
        sample,
        model_data,
        no_model_data,
        gen_data,
        teacher_model_data,
        teacher_no_model_data,
    ):
        seg = np.iinfo(np.int32).max * 2 + 1
        input_ids = np.array(sample["student_input_ids"])
        source_len = np.where(input_ids == seg)[0][0]
        prompt = input_ids[:source_len]
        input_ids = np.concatenate(
            [input_ids[:source_len], input_ids[source_len + 1 :]], axis=0
        )
        input_ids = input_ids[: self.max_length]
        input_len = len(input_ids)

        model_data["input_ids"][index][: input_len - 1] = torch.tensor(
            input_ids[:-1], dtype=torch.long
        )
        model_data["attention_mask"][index][: input_len - 1] = 1.0
        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"][index][: input_len - 1] = torch.arange(
                0, input_len - 1, dtype=torch.long
            )

        no_model_data["label"][index][: input_len - 1] = torch.tensor(
            input_ids[1:], dtype=torch.long
        )
        no_model_data["label"][index][: source_len - 1] = -100
        no_model_data["loss_mask"][index][: input_len - 1] = 1.0
        no_model_data["loss_mask"][index][: source_len - 1] = 0

        gen_data["input_ids"][index][-len(prompt) :] = torch.tensor(
            prompt, dtype=torch.long
        )
        gen_data["attention_mask"][index][-len(prompt) :] = 1.0

        for teacher in self.teacher_configs:
            teacher_id = teacher["id"]
            teacher_type = teacher["type"]
            teacher_input_ids = np.array(sample[f"teacher_{teacher_id}_input_ids"])
            teacher_source_len = np.where(teacher_input_ids == seg)[0][0]
            teacher_input_ids = np.concatenate(
                [
                    teacher_input_ids[:teacher_source_len],
                    teacher_input_ids[teacher_source_len + 1 :],
                ],
                axis=0,
            )
            teacher_input_ids = teacher_input_ids[: self.max_length]
            teacher_input_len = len(teacher_input_ids)

            teacher_model_data[teacher_id]["input_ids"][index][
                : teacher_input_len - 1
            ] = torch.tensor(teacher_input_ids[:-1], dtype=torch.long)
            teacher_model_data[teacher_id]["attention_mask"][index][
                : teacher_input_len - 1
            ] = 1.0

            if teacher_type in ["gpt2"]:
                teacher_model_data[teacher_id]["position_ids"][index][
                    : teacher_input_len - 1
                ] = torch.arange(0, teacher_input_len - 1, dtype=torch.long)

            teacher_no_model_data[teacher_id]["label"][index][
                : teacher_input_len - 1
            ] = torch.tensor(teacher_input_ids[1:], dtype=torch.long)
            teacher_no_model_data[teacher_id]["label"][index][
                : teacher_source_len - 1
            ] = -100
            teacher_no_model_data[teacher_id]["loss_mask"][index][
                : teacher_input_len - 1
            ] = 1.0
            teacher_no_model_data[teacher_id]["loss_mask"][index][
                : teacher_source_len - 1
            ] = 0

    def move_to_device(self, datazip, device):
        for data in datazip:
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)
                elif isinstance(data[key], dict):
                    for nested_key in data[key]:
                        data[key][nested_key] = data[key][nested_key].to(device)

    def collate(self, samples):
        batch_size = len(samples)
        max_length = self.max_length

        model_data = {
            "input_ids": torch.ones(batch_size, max_length, dtype=torch.long)
            * self.student_tokenizer.eos_token_id,
            "attention_mask": torch.zeros(batch_size, max_length),
        }

        if self.args.model_type in ["gpt2"]:
            model_data["position_ids"] = torch.zeros(
                batch_size, max_length, dtype=torch.long
            )

        no_model_data = {
            "label": torch.ones(batch_size, max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(batch_size, max_length),
        }

        gen_data = {
            "input_ids": torch.ones(
                batch_size, self.max_prompt_length, dtype=torch.long
            )
            * self.student_tokenizer.eos_token_id,
            "attention_mask": torch.zeros(
                batch_size, self.max_prompt_length, dtype=torch.long
            ),
        }

        teacher_model_data = {
            teacher["id"]: {
                "input_ids": torch.ones(batch_size, max_length, dtype=torch.long)
                * teacher["tokenizer"].eos_token_id,
                "attention_mask": torch.zeros(batch_size, max_length),
            }
            for teacher in self.teacher_configs
        }

        for teacher in self.teacher_configs:
            if teacher["type"] in ["gpt2"]:
                teacher_model_data[teacher["id"]]["position_ids"] = torch.zeros(
                    batch_size, max_length, dtype=torch.long
                )

        teacher_no_model_data = {
            teacher["id"]: {
                "label": torch.ones(batch_size, max_length, dtype=torch.long) * -100,
                "loss_mask": torch.zeros(batch_size, max_length),
            }
            for teacher in self.teacher_configs
        }

        for index, sample in enumerate(samples):
            self._process_lm(
                index,
                sample,
                model_data,
                no_model_data,
                gen_data,
                teacher_model_data,
                teacher_no_model_data,
            )

        for teacher_id in teacher_model_data:
            prefix = f"teacher_{teacher_id}_"
            for key in teacher_model_data[teacher_id]:
                model_data[f"{prefix}{key}"] = teacher_model_data[teacher_id][key]

            for key in teacher_no_model_data[teacher_id]:
                no_model_data[f"{prefix}{key}"] = teacher_no_model_data[teacher_id][key]

        return model_data, no_model_data, gen_data
