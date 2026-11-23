"""Verify provenance before attaching a Phase 1 adapter."""
import json

from .data import file_sha256


def attach_adapter(assistant, path, expected_sha=None):
    from peft import PeftModel

    provenance = json.loads((path / "provenance.json").read_text())
    if any(provenance[key] != assistant.config[key] for key in ("model", "revision")):
        raise ValueError("Adapter base model or revision mismatch")
    digest = file_sha256(path / "adapter_model.safetensors")
    if digest != provenance["sha256"] or (expected_sha is not None and digest != expected_sha):
        raise ValueError("Adapter weights do not match the expected SHA-256")
    assistant.model = PeftModel.from_pretrained(
        assistant.model, str(path), is_trainable=False, local_files_only=True,
    ).eval()
    return provenance
