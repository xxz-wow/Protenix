"""Counters and uncached mask validation for optional MUSA paths."""
from collections import Counter
COUNTS = Counter()
STATS = Counter()
def mask_all_ones(mask):
    return mask is None or bool((mask == 1).all().item())
def bias_all_zeros(bias):
    return bias is None or bool((bias == 0).all().item())
