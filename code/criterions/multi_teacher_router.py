import torch

from .cross_entropy_loss import CrossEntropyLoss
from .universal_logit_distillation import UniversalLogitDistillation
from .various_divergence import VariousDivergence


class MultiTeacherRouter(CrossEntropyLoss):
    def __init__(self, args, padding_id=-100) -> None:
        super(MultiTeacherRouter, self).__init__(args, padding_id=padding_id)
        self.args = args
        self.kd_rate = args.kd_rate
        self.divergence_helper = VariousDivergence(args, padding_id=padding_id)
        self.uld_helper = UniversalLogitDistillation(args, padding_id=padding_id)

    def forward(
        self,
        distiller,
        input_data,
        output_data,
        logging_output,
        batch_denom,
    ):
        self.distiller = distiller
        student_model = distiller.student_model
        outputs = student_model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None),
            output_hidden_states=True,
        )
        logits = outputs.logits
        log = {}
        loss = self.compute_cross_entropy_loss(
            outputs.logits, output_data["label"], log=log
        )[0]

        total_kd_loss = 0.0
        teacher_logits_for_report = None
        teacher_target_for_report = None

        with torch.no_grad():
            for teacher in distiller.teachers:
                teacher_model = distiller.get_teacher_model(teacher)
                teacher_model.eval()
                teacher_outputs = teacher_model(
                    input_data[distiller.get_teacher_input_key(teacher, "input_ids")],
                    attention_mask=input_data[
                        distiller.get_teacher_input_key(teacher, "attention_mask")
                    ],
                    position_ids=input_data.get(
                        distiller.get_teacher_input_key(teacher, "position_ids"), None
                    ),
                    output_hidden_states=True,
                )

                kd_loss, teacher_log = self.compute_teacher_kd_loss(
                    outputs,
                    teacher_outputs,
                    input_data,
                    output_data,
                    distiller,
                    teacher,
                )
                weighted_kd_loss = teacher["weight"] * kd_loss
                total_kd_loss += weighted_kd_loss

                teacher_id = teacher["id"]
                log[f"kd_loss_{teacher_id}"] = kd_loss
                log[f"weighted_kd_loss_{teacher_id}"] = weighted_kd_loss
                log[f"teacher_weight_{teacher_id}"] = kd_loss.new_tensor(
                    teacher["weight"]
                )
                for key, value in teacher_log.items():
                    log[f"{teacher_id}_{key}"] = value

                if teacher_logits_for_report is None:
                    teacher_logits_for_report = teacher_outputs.logits
                    teacher_target_for_report = output_data[
                        distiller.get_teacher_label_key(teacher)
                    ]

        log["kd_loss"] = total_kd_loss
        loss = (1.0 - self.kd_rate) * loss + self.kd_rate * total_kd_loss
        log["loss"] = loss
        log["accuracy"] = self.compute_token_accuracy(logits, output_data["label"])

        if self.args.report_logits and teacher_logits_for_report is not None:
            self.record_logits(
                logits,
                output_data["label"],
                log,
                teacher_logits=teacher_logits_for_report,
                teacher_target=teacher_target_for_report,
            )

        logging_output = self.record_logging_output(logging_output, batch_denom, log)
        return loss / batch_denom, logging_output

    def compute_teacher_kd_loss(
        self,
        outputs,
        teacher_outputs,
        input_data,
        output_data,
        distiller,
        teacher,
    ):
        loss_type = teacher["loss_type"]

        if loss_type in {
            "forward_kl",
            "reverse_kl",
            "adaptive_kl",
            "skewed_forward_kl",
            "skewed_reverse_kl",
            "js_divergence",
        }:
            student_logits = outputs.logits
            teacher_logits = teacher_outputs.logits
            if student_logits.shape[-1] != teacher_logits.shape[-1]:
                raise ValueError(
                    f"Loss `{loss_type}` expects matching vocab sizes, but got "
                    f"{student_logits.shape[-1]} and {teacher_logits.shape[-1]} "
                    f"for teacher `{teacher['id']}`."
                )

            loss_func = {
                "forward_kl": self.divergence_helper.compute_forward_kl_divergence,
                "reverse_kl": self.divergence_helper.compute_reverse_kl_divergence,
                "adaptive_kl": self.divergence_helper.compute_adaptive_kl_divergence,
                "skewed_forward_kl": self.divergence_helper.compute_skewed_forward_kl_divergence,
                "skewed_reverse_kl": self.divergence_helper.compute_skewed_reverse_kl_divergence,
                "js_divergence": self.divergence_helper.compute_js_divergence,
            }[loss_type]
            kd_loss = loss_func(
                student_logits,
                teacher_logits,
                output_data["label"],
            )
            return kd_loss, {}

        if loss_type in {"uld", "universal_logit_distillation"}:
            helper_log = {}
            kd_loss, helper_log = self.uld_helper.compute_universal_logit_distillation_loss(
                outputs,
                teacher_outputs,
                output_data,
                distiller,
                helper_log,
                teacher,
            )
            return kd_loss, helper_log

        raise NameError(
            f"Unsupported multi-teacher loss type `{loss_type}` for teacher "
            f"`{teacher['id']}`."
        )
