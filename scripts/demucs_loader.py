"""Demucs model loading — single source of truth for the export and validation pipelines.

The spectral-core exporter (export_spectral_core.py) and the contract
validator (validate_spectral_contract.py) both need to load pretrained Demucs
models and compute segment frames. This module owns that logic so Demucs API
changes land in one place.
"""

import gc
import math

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


def validate_weight_matrix(weights, n_models, n_sources):
    """Validate a BagOfModels weight matrix and return (matrix, per-source totals).

    ``demucs.apply.apply_model`` treats the matrix as one row per sub-model and
    one column per source, then divides each source by its total weight. Every
    source must have a positive total, otherwise the result is undefined.
    """
    if not isinstance(weights, (list, tuple)) or len(weights) != n_models:
        raise ValueError(f"expected {n_models} weight rows, got {weights!r}")
    cleaned = []
    for row in weights:
        if not isinstance(row, (list, tuple)) or len(row) != n_sources:
            raise ValueError(f"each weight row must have {n_sources} sources")
        values = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value < 0:
                raise ValueError("ensemble weights must be finite non-negative numbers")
            values.append(float(value))
        cleaned.append(values)
    totals = [sum(row[source] for row in cleaned) for source in range(n_sources)]
    if any(total <= 0 for total in totals):
        raise ValueError("every source must have a positive total ensemble weight")
    return cleaned, totals


def _validated_weights(bag, model_name):
    """Return the BagOfModels per-source weight matrix, validated.

    The shipped ``htdemucs_ft`` configuration is the 4x4 identity matrix, so
    each source must come from its own sub-model (not an average of all four).
    """
    weights = getattr(bag, "weights", None)
    try:
        normalized, _totals = validate_weight_matrix(
            weights, len(bag.models), len(bag.sources),
        )
    except ValueError as exc:
        raise RuntimeError(f"{model_name} BagOfModels weights: {exc}") from exc
    return normalized


def accumulate_weighted(acc, out, weights_row):
    """Accumulate one sub-model's ``[1, S, ...]`` output with per-source weights."""
    import numpy as np

    if out.ndim < 2 or out.shape[1] != len(weights_row):
        raise ValueError("ensemble weight row does not match the source axis")
    for value in weights_row:
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value < 0:
            raise ValueError("ensemble weights must be finite non-negative numbers")
    if acc is None:
        acc = np.zeros_like(out, dtype=np.float64)
    for source, weight in enumerate(weights_row):
        acc[:, source] += out[:, source].astype(np.float64) * weight
    return acc


def normalize_weighted(acc, weights):
    """Divide a weighted accumulator by each source's total weight."""
    if not weights:
        raise ValueError("ensemble weights must not be empty")
    _matrix, totals = validate_weight_matrix(weights, len(weights), len(weights[0]))
    if acc.ndim < 2 or acc.shape[1] != len(totals):
        raise ValueError("ensemble accumulator does not match the source axis")
    for source, total in enumerate(totals):
        acc[:, source] /= total
    return acc


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

    Loads the full BagOfModels, extracts the requested sub-model and the
    per-source weight matrix, and discards the rest to conserve memory.
    Returns ``(model, segment_frames, n_models, weights)``.
    """
    from demucs.pretrained import get_model

    bag = get_model(model_name)
    if not hasattr(bag, "models"):
        raise RuntimeError(f"{model_name} is not a BagOfModels")

    n_models = len(bag.models)
    if index >= n_models:
        raise RuntimeError(f"Sub-model index {index} >= {n_models}")

    weights = _validated_weights(bag, model_name)
    model = bag.models[index]
    del bag
    gc.collect()

    model = _prep(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded sub-model {index}: {total_params:,} parameters")
    return model, _segment_frames(model), n_models, weights


def load_bag_weights(model_name):
    """Load only the per-source BagOfModels weight matrix (no sub-model kept)."""
    from demucs.pretrained import get_model

    bag = get_model(model_name)
    if not hasattr(bag, "models"):
        raise RuntimeError(f"{model_name} is not a BagOfModels")
    weights = _validated_weights(bag, model_name)
    del bag
    gc.collect()
    return weights


def iter_sub_models(model_name):
    """Yield sub-models and the ensemble weights of a BagOfModels.

    Loads the bag once, yields sub-models one at a time. Caller is responsible
    for freeing each sub-model between iterations to keep memory low.
    Returns ``(n_models, segment_frames, weights, iterator)``; each iterator
    item is ``(sub_model, segment_frames, index)``.
    """
    from demucs.pretrained import get_model

    bag = get_model(model_name)
    if not hasattr(bag, "models"):
        raise RuntimeError(f"{model_name} is not a BagOfModels")

    n_models = len(bag.models)
    seg = _segment_frames(bag.models[0])
    weights = _validated_weights(bag, model_name)
    models = list(bag.models)
    del bag
    gc.collect()

    def gen():
        for i, sub_model in enumerate(models):
            yield _prep(sub_model), seg, i

    return n_models, seg, weights, gen()
