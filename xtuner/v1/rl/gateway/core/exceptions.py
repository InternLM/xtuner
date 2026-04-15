class GatewayError(RuntimeError):
    """Base exception for gateway failures."""


class GatewayStateError(GatewayError):
    """Raised when the gateway app is missing required runtime state."""


class ModelNotFoundError(GatewayError):
    """Raised when a requested model is not exposed by the backend."""

    def __init__(self, model: str):
        super().__init__(f"Model '{model}' is not available.")
        self.model = model


class ContextLengthExceededError(GatewayError):
    """Raised when the prompt is too long for the model's context window."""

    def __init__(self, prompt_tokens: int, context_length: int):
        super().__init__(f"Input is too long: prompt_tokens={prompt_tokens}, context_length={context_length}.")
        self.prompt_tokens = prompt_tokens
        self.context_length = context_length
