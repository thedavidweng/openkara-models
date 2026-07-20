"""Demucs model loading — single source of truth for convert and validate pipelines.

Both the conversion pipeline (convert_htdemucs_to_onnx.py) and the validation
pipeline (validate_onnx.py) need to load pretrained Demucs models and compute
segment frames. This module owns that logic so Demucs API changes land in one
place.
"""

import gc

SUPPORTED_MODELS = ("htdemucs", "htdemucs_ft")


def _unwrap_bag(bag):
    if hasattr(bag, "models"):
        model = bag.models[0]
        print(f"Unwrapped BagOfModels → {type(model).__name__}")
    else:
        model = bag
    return model


def _prep(model):
    model.eval()
    model.cpu()
    return model


def _segment_frames(model):
    return int(model.segment * model.samplerate)


def load(model_name):
    """Load a single pretrained Demucs model, unwrapped from BagOfModels.

    Returns (model, segment_frames).
    """
    from demucs.pretrained import get_model

    model = _prep(_unwrap_bag(get_model(model_name)))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {model_name} model: {total_params:,} parameters")

    seg = _segment_frames(model)
    print(f"Model segment: {model.segment}s = {seg} frames")
    return model, seg


def load_sub_model(model_name, index):
    """Load a single sub-model from a BagOfModels by index.

    Loads the full BagOfModels, extracts the requested sub-model, and discards
    the rest to conserve memory. Returns (model, segment_frames, n_models).
    """
    from demucs.pretrained import get_model

    bag = get_model(model_name)
    if not hasattr(bag, "models"):
        raise RuntimeError(f"{model_name} is not a BagOfModels")

    n_models = len(bag.models)
    if index >= n_models:
        raise RuntimeError(f"Sub-model index {index} >= {n_models}")

    model = bag.models[index]
    del bag
    gc.collect()

    model = _prep(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded sub-model {index}: {total_params:,} parameters")
    return model, _segment_frames(model), n_models


def iter_sub_models(model_name):
    """Yield (sub_model, segment_frames) for each sub-model in a BagOfModels.

    Loads the bag once, yields sub-models one at a time. Caller is responsible
    for freeing each sub-model between iterations to keep memory low.
    Returns (n_models, iterator) so the caller knows the count up front.
    """
    from demucs.pretrained import get_model

    bag = get_model(model_name)
    if not hasattr(bag, "models"):
        raise RuntimeError(f"{model_name} is not a BagOfModels")

    n_models = len(bag.models)
    seg = _segment_frames(bag.models[0])
    models = list(bag.models)
    del bag
    gc.collect()

    def gen():
        for i, sub_model in enumerate(models):
            yield _prep(sub_model), seg, i

    return n_models, seg, gen()
