from sf_examples.minihack.models.scaled import ScaledNet

MODELS = [
    ScaledNet,
]
MODELS_LOOKUP = {c.__name__: c for c in MODELS}
