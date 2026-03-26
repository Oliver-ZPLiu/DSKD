import random
import torch
import os
from torch.utils.data import Dataset

from torch.distributed import get_rank, get_world_size
from utils import print_rank, log_rank
from tqdm import tqdm
import json
from data_utils.sample_formats import normalize_sft_records


class PromptDataset(Dataset):
    def __init__(self, args, tokenizer, split, data_path=None, num=-1):
        super().__init__()
        self.tokenizer = tokenizer

        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.pad_id = self.tokenizer.eos_token_id
        self.max_length = args.max_length
        self.min_prompt_length = args.min_prompt_length
        self.max_prompt_length = args.max_prompt_length

        if args.json_data:
            self.data, self.origin_data = self.load_data_json(data_path, num)
            self.raw = self.origin_data
            self.answers = [x["references"] for x in self.raw]
        else:
            # txt data
            self.data = self.load_data_txt(data_path)
            self.raw = []
            self.answers = []

        self.num = len(self.data)
        # self.data = self.data[:self.num]

        if not self.answers:
            log_rank("WARNING: No answers exist")

        self.label_map = {}
        for refs in self.answers:
            if not refs:
                continue
            token_ids = tokenizer.encode(refs[0], add_special_tokens=False)
            if len(token_ids) == 0:
                continue
            self.label_map[token_ids[0]] = refs[0]
            
        
        log_rank(f"Num instances: {len(self.data)}")
            
    def __len__(self):
        return self.num

    def load_data_json(self, data_path, data_num):
        if os.path.exists(os.path.join(data_path, f"{self.split}_{self.args.model_type}.jsonl")):
            data_path = os.path.join(data_path, f"{self.split}_{self.args.model_type}.jsonl")
        else:
            data_path = os.path.join(data_path, f"{self.split}.jsonl")
        
        with open(data_path) as f:
            lines = f.readlines()
        data_origin = [json.loads(line) for line in lines]
        data_origin = normalize_sft_records(data_origin)
        data_origin = data_origin[:data_num] if data_num != -1 else data_origin

        show_progress = True
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            show_progress = get_rank() == 0

        data = []
        for d in tqdm(data_origin, desc="Loading Data ", disable=(not show_progress)):
            prompt = d["prompt"].replace("<n>", "\n")
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            output_ids = self.tokenizer.encode(d["output"], add_special_tokens=False)
            output_ids += [self.tokenizer.eos_token_id]
            data.append({
                "prompt_ids": prompt_ids,
                "output_ids": output_ids[:self.max_length - len(prompt_ids)]
            })
        log_rank("Load End")
        return data, data_origin

    def load_data_txt(self, data_path):
        with open(os.path.join(data_path, f"{self.split}.txt")) as f:
            lines = f.readlines()
        data = []
        log_rank("Loading Data")
        for line in lines:
            line = line.strip()
            line = line.replace("<n>", "\n")
            prompt = self.tokenizer.encode(line)
            data.append(prompt)
        log_rank("Load End")
        return data

    def verbalizer(self):
        return self.label_map

    def __getitem__(self, index: int):
        data = self.data[index]
        if self.args.bin_data:
            data = data.astype(int)
        elif self.args.json_data:
            output_ids = data["output_ids"]
            data = data["prompt_ids"]
        
        prompt_length = self.max_prompt_length

        prompt = data[:prompt_length]
        rest = data[prompt_length:]  
        if self.args.json_data:
            if output_ids is not None:
                rest = output_ids  
    
        return index, prompt, rest
    
    def collate(self, samples):
        bs = len(samples)
        
        max_prompt_length = self.max_prompt_length
        max_rest_length = max([len(samp[2]) for samp in samples])
        
        model_batch = {
            "input_ids": torch.ones(bs, max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_prompt_length, dtype=torch.long),
            # "position_ids": torch.zeros(bs, max_prompt_length, dtype=torch.long)
        }
        
        no_model_batch = {
            "idx": torch.zeros(bs, dtype=torch.long),
            "rest_ids": torch.ones(bs, max_rest_length, dtype=torch.long) * self.pad_id
        }
        
        for i, (idx, prompt, rest) in enumerate(samples):
            # left padding
            model_batch["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            model_batch["attention_mask"][i][-len(prompt):] = 1
            # model_batch["input_ids"][i][:len(prompt)] = torch.tensor(prompt, dtype=torch.long)
            # model_batch["input_ids"][i][len(prompt):len(prompt)+len(rest)] = torch.tensor(rest, dtype=torch.long)
            # model_batch["attention_mask"][i][:len(prompt)+len(rest)] = 1
            # model_batch["prompt_ids"] = torch.tensor([prompt], dtype=torch.long)
            # print(model_batch["input_ids"])
            # model_batch["position_ids"][i][-len(prompt):] = torch.arange(len(prompt))
            no_model_batch["idx"][i] = idx
            no_model_batch["rest_ids"][i][:len(rest)] = torch.tensor(rest, dtype=torch.long)
        
        return model_batch, no_model_batch

    def move_to_device(self, model_batch, no_model_batch, device):
        for k in model_batch:
            model_batch[k] = model_batch[k].to(device)        
        for k in no_model_batch:
            no_model_batch[k] = no_model_batch[k].to(device)    
        
        return model_batch, no_model_batch
