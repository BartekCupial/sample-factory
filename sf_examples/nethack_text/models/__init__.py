from sf_examples.nethack_text.models.huggingface import HuggingFace
from sf_examples.nethack_text.models.vllm import VLLM

MODELS = [
    HuggingFace,
    VLLM,
]
MODELS_LOOKUP = {c.__name__: c for c in MODELS}
