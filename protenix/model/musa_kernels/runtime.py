"""Optional counters for the MUSA paths implemented in model functions."""
from collections import Counter
COUNTS = Counter()
STATS = Counter()

# Keep mask validation on CPU after its first reduction. Tensor identity and
# mutation version prevent reuse for a different allocation or an edited mask.
import weakref
_MASK_RESULTS = {}
def _cached_mask_result(tensor, expected):
    if tensor is None:
        STATS["mask_none_fastpath"] += 1
        return True
    try:
        version = tensor._version
    except RuntimeError:
        STATS["mask_cache_uncacheable"] += 1
        return bool((tensor == expected).all().item())
    key = (id(tensor), expected)
    signature = (version, tensor.data_ptr(), tuple(tensor.shape), tensor.stride(),
                 tensor.device, tensor.dtype)
    entry = _MASK_RESULTS.get(key)
    if entry is not None and entry[0]() is tensor and entry[1] == signature:
        STATS["mask_cache_hits"] += 1
        return entry[2]
    value = bool((tensor == expected).all().item())
    if len(_MASK_RESULTS) >= 256:
        _MASK_RESULTS.clear()
    _MASK_RESULTS[key] = (weakref.ref(tensor), signature, value)
    STATS["mask_cache_misses"] += 1
    return value
def mask_all_ones(mask):
    return _cached_mask_result(mask, 1)
def bias_all_zeros(bias):
    return _cached_mask_result(bias, 0)
