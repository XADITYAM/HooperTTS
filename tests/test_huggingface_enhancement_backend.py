from core.enhancement_backends import (
    HuggingFaceEnhancementBackend,
    HuggingFaceEnhancementConfig,
)
from core.script_enhancement import EnhancementMode, ScriptEnhancer, ScriptIntelligence


class FakeInputIds:
    shape = (1, 4)


class FakeInputs(dict):
    def __init__(self) -> None:
        super().__init__(input_ids=FakeInputIds())

    def to(self, device: object) -> "FakeInputs":
        self.device = device
        return self


class FakeTokenizer:
    eos_token_id = 0

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[object] = []

    def apply_chat_template(self, messages, **kwargs):
        self.prompts.append(messages)
        return FakeInputs()

    def decode(self, output_ids, skip_special_tokens: bool) -> str:
        return self.response


class FakeModel:
    device = "cuda:0"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [[1, 2, 3, 4, 5]]


def make_backend(response: str = "Grand Theft Auto 6 arrives on August 27."):
    calls: list[tuple[str, str]] = []
    tokenizer = FakeTokenizer(response)
    model = FakeModel()

    def tokenizer_loader(model_id: str):
        calls.append(("tokenizer", model_id))
        return tokenizer

    def model_loader(model_id: str, **kwargs):
        calls.append(("model", model_id))
        assert kwargs["device_map"] == "auto"
        assert kwargs["torch_dtype"] == "auto"
        return model

    backend = HuggingFaceEnhancementBackend(
        config=HuggingFaceEnhancementConfig(model_id="Qwen/Qwen3-1.7B"),
        tokenizer_loader=tokenizer_loader,
        model_loader=model_loader,
    )
    backend._check_resources = lambda: None
    return backend, calls, tokenizer, model


def test_backend_is_lazy_and_releases_model_after_generation() -> None:
    backend, calls, tokenizer, model = make_backend()

    assert calls == []
    result = ScriptEnhancer(backend=backend).enhance(
        "Grand Theft Auto 6 arrives on August 27.", profile="friendslop_gaming"
    )

    assert calls == [
        ("tokenizer", "Qwen/Qwen3-1.7B"),
        ("model", "Qwen/Qwen3-1.7B"),
    ]
    assert result.backend_available
    assert not backend.is_loaded
    assert tokenizer.prompts
    assert model.calls


def test_backend_output_is_validated_and_creates_change_records() -> None:
    backend, _, _, _ = make_backend(
        "Grand Theft Auto 6 arrives on August 27, so the wait is almost over. "
        "Players explore Vice City."
    )
    source = "Grand Theft Auto 6 arrives on August 27. Players explore Vice City."

    result = ScriptEnhancer(backend=backend).enhance(
        source, profile="friendslop_gaming"
    )

    assert result.validation.passed
    assert result.enhanced_text != source
    assert len(result.changes) == 1
    assert result.changes[0].sentence_index == 0


def test_empty_model_output_falls_back_to_original() -> None:
    backend, _, _, _ = make_backend("")
    source = "Grand Theft Auto 6 arrives on August 27."

    result = ScriptEnhancer(backend=backend).enhance(source)

    assert result.enhanced_text == source
    assert result.backend_available
    assert result.changes == ()
    assert "no usable script" in result.diagnostic.lower()


def test_hugging_face_candidate_still_uses_protected_span_validation() -> None:
    backend, _, _, _ = make_backend("A new game arrives next year.")
    source = "Grand Theft Auto 6 arrives on August 27 for PS5."

    result = ScriptEnhancer(backend=backend).enhance(source)

    assert result.enhanced_text == source
    assert not result.validation.passed
    assert "rejected" in result.diagnostic.lower()


def test_model_load_failure_returns_useful_diagnostic() -> None:
    def failing_loader(model_id: str):
        raise OSError("offline")

    backend = HuggingFaceEnhancementBackend(
        tokenizer_loader=failing_loader,
        model_loader=lambda *args, **kwargs: object(),
    )
    backend._check_resources = lambda: None

    result = ScriptEnhancer(backend=backend).enhance("Grand Theft Auto 6 arrives.")

    assert not result.backend_available
    assert result.enhanced_text == "Grand Theft Auto 6 arrives."
    assert "Qwen/Qwen3-1.7B" in result.diagnostic
    assert "offline" in result.diagnostic


def test_model_identifier_is_configurable_without_code_changes() -> None:
    seen: list[str] = []
    backend = HuggingFaceEnhancementBackend(
        config=HuggingFaceEnhancementConfig(model_id="Qwen/Qwen3-0.6B"),
        tokenizer_loader=lambda model_id: seen.append(model_id) or FakeTokenizer(""),
        model_loader=lambda model_id, **kwargs: FakeModel(),
    )
    backend._check_resources = lambda: None

    ScriptEnhancer(backend=backend).enhance("Grand Theft Auto 6 arrives.")

    assert seen == ["Qwen/Qwen3-0.6B"]


def test_environment_configuration_can_select_smaller_model() -> None:
    import os

    previous = os.environ.get("HOOPERTTS_ENHANCEMENT_MODEL_ID")
    try:
        os.environ["HOOPERTTS_ENHANCEMENT_MODEL_ID"] = "Qwen/Qwen3-0.6B"
        config = HuggingFaceEnhancementConfig.from_environment()
    finally:
        if previous is None:
            os.environ.pop("HOOPERTTS_ENHANCEMENT_MODEL_ID", None)
        else:
            os.environ["HOOPERTTS_ENHANCEMENT_MODEL_ID"] = previous

    assert config.model_id == "Qwen/Qwen3-0.6B"


def test_optimize_only_does_not_load_hugging_face_backend() -> None:
    backend, calls, _, _ = make_backend()
    intelligence = ScriptIntelligence(enhancer=ScriptEnhancer(backend=backend))

    result = intelligence.process(
        "Imagine opening GTA 6.",
        mode=EnhancementMode.OPTIMIZE_ONLY,
        profile="youtube_shorts",
    )

    assert result.enhancement is None
    assert calls == []


def test_enhance_and_optimize_invokes_hugging_face_backend() -> None:
    backend, calls, _, _ = make_backend()
    intelligence = ScriptIntelligence(enhancer=ScriptEnhancer(backend=backend))

    result = intelligence.process(
        "Grand Theft Auto 6 arrives on August 27.",
        mode=EnhancementMode.ENHANCE_AND_OPTIMIZE,
        profile="friendslop_gaming",
    )

    assert result.enhancement is not None
    assert result.enhancement.backend_available
    assert calls


if __name__ == "__main__":
    test_backend_is_lazy_and_releases_model_after_generation()
    test_backend_output_is_validated_and_creates_change_records()
    test_empty_model_output_falls_back_to_original()
    test_hugging_face_candidate_still_uses_protected_span_validation()
    test_model_load_failure_returns_useful_diagnostic()
    test_model_identifier_is_configurable_without_code_changes()
    test_environment_configuration_can_select_smaller_model()
    test_optimize_only_does_not_load_hugging_face_backend()
    test_enhance_and_optimize_invokes_hugging_face_backend()
