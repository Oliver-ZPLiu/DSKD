from typing import Any, Dict, List, Tuple


ROLE_NAME_MAP = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool",
}


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _normalize_role(role: Any) -> str:
    text = _to_text(role).strip().lower()
    if text in ROLE_NAME_MAP:
        return ROLE_NAME_MAP[text]
    if not text:
        return "Unknown"
    return text.capitalize()


def _render_messages(messages: List[Dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = _to_text(message.get("content", "")).strip()
        if not content:
            continue
        role = _normalize_role(message.get("role", ""))
        lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def _build_prompt_from_instruction(record: Dict[str, Any]) -> str:
    instruction = _to_text(record.get("instruction", "")).strip()
    input_text = _to_text(record.get("input", "")).strip()
    if instruction and input_text:
        return f"{instruction}\n{input_text}"
    return instruction or input_text


def _normalize_output(output: Any) -> Tuple[str, List[str]]:
    if isinstance(output, list):
        refs = [_to_text(x).strip() for x in output]
        refs = [x for x in refs if x]
        if refs:
            return refs[0], refs
        return "", []
    text = _to_text(output).strip()
    if text:
        return text, [text]
    return "", []


def _convert_messages_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = record.get("messages", None)
    if not isinstance(messages, list):
        return []

    samples = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = _to_text(message.get("role", "")).strip().lower()
        if role != "assistant":
            continue

        output = _to_text(message.get("content", "")).strip()
        if not output:
            continue

        prompt = _render_messages(messages[:idx])
        if prompt:
            prompt = f"{prompt}\nAssistant:"
        else:
            fallback_prompt = _to_text(record.get("prompt", "")).strip()
            prompt = "Assistant:"
            if fallback_prompt:
                prompt = f"User: {fallback_prompt}\nAssistant:"

        samples.append({
            "prompt": prompt,
            "output": output,
            "references": [output],
        })

    return samples


def normalize_sft_records(raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for record in raw_records:
        if not isinstance(record, dict):
            continue

        message_samples = _convert_messages_record(record)
        if message_samples:
            normalized.extend(message_samples)
            continue

        prompt = _to_text(record.get("prompt", "")).strip()
        if not prompt:
            prompt = _build_prompt_from_instruction(record).strip()

        output, references = _normalize_output(
            record.get("output", record.get("response", ""))
        )
        if not references:
            references = [output]

        normalized.append({
            "prompt": prompt,
            "output": output,
            "references": references,
        })

    return normalized
