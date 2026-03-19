import json
import os

import torch
import torch.nn as nn
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from utils import log_rank


class Distiller(nn.Module):
    def __init__(self, args, device):
        super(Distiller, self).__init__()
        self.args = args
        self.device = device
        self.student_model_type = args.model_type
        self.student_model, self.student_tokenizer = self.load_student_model()

        self.teachers = []
        self.teacher_model = None
        self.teacher_id = None
        self.teacher_model_type = args.teacher_model_type
        self.teacher_tokenizers = {}
        self.teacher_models = {}
        self.teacher_weights = []
        self.teacher_ids = []
        self.teacher_model_types = []
        self.teacher_loss_types = []
        self.additional_teacher_models = []

        if self.args.teacher_model_path is not None:
            self.teachers = self.build_teacher_configs()
            for teacher in self.teachers:
                teacher_model, teacher_tokenizer = self.load_teacher_model(
                    teacher["path"],
                    teacher["type"],
                    teacher.get("peft_path"),
                )
                teacher["model"] = teacher_model
                teacher["tokenizer"] = teacher_tokenizer

            self.teacher_model = self.teachers[0]["model"]
            self.teacher_id = self.teachers[0]["id"]
            self.teacher_model_type = self.teachers[0]["type"]
            self.teacher_tokenizers = {
                teacher["id"]: teacher["tokenizer"] for teacher in self.teachers
            }
            self.teacher_models = {
                teacher["id"]: teacher["model"] for teacher in self.teachers
            }
            self.teacher_ids = [teacher["id"] for teacher in self.teachers]
            self.teacher_model_types = [teacher["type"] for teacher in self.teachers]
            self.teacher_loss_types = [teacher["loss_type"] for teacher in self.teachers]
            self.teacher_weights = [teacher["weight"] for teacher in self.teachers]
            self.additional_teacher_models = [
                teacher["model"] for teacher in self.teachers[1:]
            ]
            log_rank(
                "Teacher configs: {}".format(
                    [
                        {
                            "id": teacher["id"],
                            "type": teacher["type"],
                            "loss_type": teacher["loss_type"],
                            "weight": teacher["weight"],
                        }
                        for teacher in self.teachers
                    ]
                )
            )

        if self.teacher_model and args.projector_config_path:
            self.set_and_load_existing_projectors()
            log_rank(f"projector structure: {self.projectors}")

        if args.teacher_to_student_token_mapping is not None:
            self.tea2stu_token_mapping = json.load(
                open(args.teacher_to_student_token_mapping)
            )
            log_rank(
                f"Load teacher-to-student token mapping from "
                f"{args.teacher_to_student_token_mapping}"
            )

        if args.teacher_to_student_id_mapping is not None:
            self.tea2stu_id_mapping = json.load(
                open(args.teacher_to_student_id_mapping)
            )
            log_rank(
                f"Load teacher-to-student id mapping from "
                f"{args.teacher_to_student_id_mapping}"
            )

            self.stu2tea_id_mapping = {}
            for tea_id in self.tea2stu_id_mapping:
                stu_id = self.tea2stu_id_mapping[tea_id]
                if stu_id not in self.stu2tea_id_mapping:
                    self.stu2tea_id_mapping[stu_id] = [int(tea_id)]
                else:
                    self.stu2tea_id_mapping[stu_id].append(int(tea_id))

            max_align_num = 1
            for stu_id in self.stu2tea_id_mapping:
                self.stu2tea_id_mapping[stu_id] = (
                    self.stu2tea_id_mapping[stu_id][:max_align_num]
                    + [self.stu2tea_id_mapping[stu_id][-1]]
                    * max(0, max_align_num - len(self.stu2tea_id_mapping[stu_id]))
                )

            self.tea2stu_id_mapping = torch.LongTensor(
                list(self.tea2stu_id_mapping.values())
            ).to(device)
            self.stu2tea_id_mapping_tea = torch.LongTensor(
                list(self.stu2tea_id_mapping.values())
            ).to(device)
            self.stu2tea_id_mapping_stu = torch.LongTensor(
                list(self.stu2tea_id_mapping.keys())
            ).to(device)

    @staticmethod
    def add_distiller_args(parser):
        group = parser.add_argument_group("distiller", "distiller configurations")
        group.add_argument(
            "--projector-config-path",
            type=str,
            default=None,
            help="path to projector_config.json",
        )
        group.add_argument(
            "--projector-path",
            type=str,
            default=None,
            help="path to pretrained projector",
        )
        group.add_argument(
            "--projector-lr",
            type=float,
            default=0.001,
            help="learning rate only for projection",
        )
        group.add_argument(
            "--pretrained-projector",
            type=str,
            default=None,
            help="pretrained projector name",
        )
        group.add_argument(
            "--pretrained-projector-lr",
            type=float,
            default=0.001,
            help="learning rate only for pretrained projector",
        )
        group.add_argument(
            "--vocab-alignment-path",
            type=str,
            default=None,
            help="path for the vocab alignment file",
        )
        group.add_argument(
            "--teacher-to-student-token-mapping",
            type=str,
            default=None,
            help="path for the vocab alignment file (token, teacher-to-student)",
        )
        group.add_argument(
            "--teacher-to-student-id-mapping",
            type=str,
            default=None,
            help="path for the vocab alignment file (id, teacher-to-student)",
        )
        group.add_argument(
            "--student-to-teacher-token-mapping",
            type=str,
            default=None,
            help="path for the vocab alignment file (token, student-to-teacher)",
        )
        group.add_argument(
            "--student-to-teacher-id-mapping",
            type=str,
            default=None,
            help="path for the vocab alignment file (id, student-to-teacher)",
        )
        return parser

    def _split_csv(self, raw_value):
        if raw_value is None:
            return []
        return [item.strip() for item in raw_value.split(",") if item.strip()]

    def build_teacher_configs(self):
        main_teacher_type = self.args.teacher_model_type or self.args.model_type
        main_teacher_id = self.args.teacher_id or "teacher0"
        main_loss_type = self.args.teacher_loss_type or self.args.kd_objective

        teacher_configs = [
            {
                "id": main_teacher_id,
                "path": self.args.teacher_model_path,
                "type": main_teacher_type,
                "loss_type": main_loss_type,
                "peft_path": self.args.teacher_peft_path,
            }
        ]

        additional_paths = self._split_csv(self.args.additional_teacher_paths)
        additional_types = self._split_csv(self.args.additional_teacher_types)
        additional_ids = self._split_csv(self.args.additional_teacher_ids)
        additional_loss_types = self._split_csv(
            self.args.additional_teacher_loss_types
        )
        additional_peft_paths = self._split_csv(
            self.args.additional_teacher_peft_paths
        )

        for index, path in enumerate(additional_paths):
            teacher_configs.append(
                {
                    "id": additional_ids[index]
                    if index < len(additional_ids)
                    else f"teacher{index + 1}",
                    "path": path,
                    "type": additional_types[index]
                    if index < len(additional_types)
                    else self.args.model_type,
                    "loss_type": additional_loss_types[index]
                    if index < len(additional_loss_types)
                    else main_loss_type,
                    "peft_path": additional_peft_paths[index]
                    if index < len(additional_peft_paths)
                    else None,
                }
            )

        teacher_ids = [teacher["id"] for teacher in teacher_configs]
        if len(set(teacher_ids)) != len(teacher_ids):
            raise ValueError(
                f"Teacher ids must be unique, but got duplicates: {teacher_ids}"
            )

        if self.args.multi_teacher_weights is not None:
            weights = [
                float(weight.strip())
                for weight in self.args.multi_teacher_weights.split(",")
                if weight.strip()
            ]
            if len(weights) != len(teacher_configs):
                raise ValueError(
                    "The number of multi-teacher weights must match the number "
                    "of teachers."
                )
        else:
            weights = [1.0 / len(teacher_configs)] * len(teacher_configs)

        for teacher, weight in zip(teacher_configs, weights):
            teacher["weight"] = weight

        return teacher_configs

    def get_teacher(self, teacher=None, teacher_id=None, index=None):
        if isinstance(teacher, dict):
            return teacher

        if isinstance(teacher, str):
            teacher_id = teacher

        if isinstance(teacher, int):
            index = teacher

        if teacher_id is not None:
            for teacher_cfg in self.teachers:
                if teacher_cfg["id"] == teacher_id:
                    return teacher_cfg
            raise KeyError(f"Unknown teacher id: {teacher_id}")

        if index is not None:
            return self.teachers[index]

        if not self.teachers:
            raise ValueError("No teacher has been configured.")
        return self.teachers[0]

    def get_teacher_prefix(self, teacher=None, teacher_id=None, index=None):
        teacher_cfg = self.get_teacher(teacher, teacher_id, index)
        return f'teacher_{teacher_cfg["id"]}_'

    def get_teacher_input_key(self, teacher, key):
        return f"{self.get_teacher_prefix(teacher)}{key}"

    def get_teacher_label_key(self, teacher):
        return f"{self.get_teacher_prefix(teacher)}label"

    def get_teacher_model(self, teacher=None, teacher_id=None, index=None):
        teacher_cfg = self.get_teacher(teacher, teacher_id, index)
        return teacher_cfg["model"]

    def get_teacher_tokenizer(self, teacher=None, teacher_id=None, index=None):
        teacher_cfg = self.get_teacher(teacher, teacher_id, index)
        return teacher_cfg["tokenizer"]

    def load_tokenizer(self, model_type, path):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if model_type in [
            "gpt2",
            "opt",
            "llama",
            "gptj",
            "llama2",
            "mistral",
            "tinyllama",
            "minicpm",
        ]:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif model_type == "qwen":
            tokenizer.eos_token_id = 151643
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer

    def set_and_load_existing_projectors(self):
        self.projectors = nn.ModuleDict()
        projector_config = json.load(open(self.args.projector_config_path))
        name_dict = {
            "s": self.student_hidden_size,
            "t": self.teacher_hidden_size,
            "relu": nn.ReLU(),
        }
        for projector_name in projector_config:
            if projector_config[projector_name]["enabled"]:
                self.projectors[projector_name] = nn.Sequential()

                structure = projector_config[projector_name]["structure"].split("-")
                for index in range(len(structure)):
                    if structure[index] not in ["relu"]:
                        coef = (
                            1
                            if not len(structure[index][:-1])
                            else int(structure[index][:-1])
                        )
                        base_size = name_dict[structure[index][-1]]
                        structure[index] = coef * base_size

                for index in range(len(structure) - 1):
                    if isinstance(structure[index], int) and isinstance(
                        structure[index + 1], int
                    ):
                        self.projectors[projector_name].append(
                            nn.Linear(structure[index], structure[index + 1])
                        )
                    elif isinstance(structure[index], int) and isinstance(
                        structure[index + 1], str
                    ):
                        self.projectors[projector_name].append(
                            name_dict[structure[index + 1]]
                        )
                        last_size = structure[index]
                    elif isinstance(structure[index], str) and isinstance(
                        structure[index + 1], int
                    ):
                        self.projectors[projector_name].append(
                            nn.Linear(last_size, structure[index + 1])
                        )
                    else:
                        raise NotImplementedError(
                            f"Invalid structure for '{structure}'"
                        )

        self.load_existing_projectors()

    def load_existing_projectors(self):
        if self.args.projector_path is not None:
            projector_path = os.path.join(self.args.projector_path, "projector.pt")
        else:
            projector_path = os.path.join(self.args.model_path, "projector.pt")

        if os.path.exists(projector_path):
            projector_params = torch.load(
                projector_path, map_location=f"cuda:{self.device}"
            )
            log_rank(
                "Existing projector params: {}".format(list(projector_params.keys()))
            )
            for key in self.projectors:
                try:
                    state_dict = {
                        name.split(".", 1)[1]: projector_params[name]
                        for name in projector_params
                        if name.startswith(key)
                    }
                    self.projectors[key].load_state_dict(state_dict)
                    log_rank(f"Load projector '{key}' from current path.")
                except Exception:
                    log_rank(f"Not compatible for projector '{key}'")
                    continue

    def load_student_model(self):
        log_rank("Loading student model...")
        config = AutoConfig.from_pretrained(
            self.args.model_path, trust_remote_code=True
        )
        config.is_model_parallel = False

        tokenizer = self.load_tokenizer(self.args.model_type, self.args.model_path)

        if hasattr(config, "n_embed"):
            self.student_hidden_size = config.n_embed
        else:
            self.student_hidden_size = config.hidden_size

        if self.args.model_dtype == "fp32":
            self.dtype = torch.float32
        elif self.args.model_dtype == "bf16":
            self.dtype = torch.bfloat16
        elif self.args.model_dtype == "fp16":
            self.dtype = torch.float16
        else:
            raise NotImplementedError(
                f"Invalid model_dtype for `{self.args.model_dtype}`"
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.args.model_path,
            config=config,
            device_map=None,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )

        if self.args.peft is not None:
            if self.args.peft == "lora":
                model.enable_input_require_grads()
                if self.args.peft_path is not None:
                    if self.args.do_train:
                        _model = PeftModel.from_pretrained(model, self.args.peft_path)
                        state_dict = dict(_model.state_dict().items())
                        peft_config = LoraConfig(
                            task_type=TaskType.CAUSAL_LM,
                            inference_mode=(not self.args.do_train),
                            r=self.args.peft_lora_r,
                            lora_alpha=self.args.peft_lora_alpha,
                            lora_dropout=self.args.peft_lora_dropout,
                        )
                        model = get_peft_model(model, peft_config)
                        model.load_state_dict(state_dict)
                        del _model
                        del state_dict
                    else:
                        model = PeftModel.from_pretrained(model, self.args.peft_path)
                else:
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=(not self.args.do_train),
                        r=self.args.peft_lora_r,
                        lora_alpha=self.args.peft_lora_alpha,
                        lora_dropout=self.args.peft_lora_dropout,
                    )
                    model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
            else:
                raise NotImplementedError
        else:
            log_rank(
                " > number of parameters: {:,}".format(
                    sum([param.nelement() for param in model.parameters()])
                )
            )

        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        return model, tokenizer

    def load_teacher_model(self, teacher_path, teacher_type, teacher_peft_path=None):
        log_rank(f"Loading teacher model from {teacher_path} (type: {teacher_type})...")
        config = AutoConfig.from_pretrained(teacher_path)
        config.is_model_parallel = False

        tokenizer = self.load_tokenizer(teacher_type, teacher_path)

        if hasattr(config, "n_embed"):
            teacher_hidden_size = config.n_embed
        else:
            teacher_hidden_size = config.hidden_size

        if not hasattr(self, "teacher_hidden_size"):
            self.teacher_hidden_size = teacher_hidden_size

        model = AutoModelForCausalLM.from_pretrained(
            teacher_path,
            config=config,
            device_map=None,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )

        if self.args.peft is not None and teacher_peft_path is not None:
            if self.args.peft == "lora":
                model = PeftModel.from_pretrained(model, teacher_peft_path)
                model = model.merge_and_unload()
            else:
                raise NotImplementedError
        else:
            log_rank(
                " > number of parameters of the teacher model: {:,}".format(
                    sum([param.nelement() for param in model.parameters()])
                )
            )

        for param in model.parameters():
            param.requires_grad = False
        return model, tokenizer

    def add_optimizer_param_group(self, optimizer):
        if hasattr(self, "projectors"):
            if self.args.projector_lr:
                pretrained_proj = (
                    self.args.pretrained_projector.split(",")
                    if self.args.pretrained_projector is not None
                    else []
                )
                optimizer.add_param_group(
                    {
                        "params": [
                            param
                            for block in self.projectors
                            if block not in pretrained_proj
                            for param in self.projectors[block].parameters()
                        ],
                        "lr": self.args.projector_lr,
                    }
                )
                optimizer.add_param_group(
                    {
                        "params": [
                            param
                            for block in self.projectors
                            if block in pretrained_proj
                            for param in self.projectors[block].parameters()
                        ],
                        "lr": self.args.pretrained_projector_lr,
                    }
                )
            else:
                optimizer.add_param_group(
                    {
                        "params": [
                            param
                            for block in self.projectors
                            for param in self.projectors[block].parameters()
                        ],
                    }
                )
        return optimizer

    def forward(self, criterion, batch, logging_output, loss_denom):
        input_data = batch["input_batch"]
        output_data = batch["output_batch"]
        loss, logging_output = criterion(
            self,
            input_data,
            output_data,
            logging_output,
            loss_denom,
        )
        return loss, logging_output
