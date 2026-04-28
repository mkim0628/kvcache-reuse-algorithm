import random
from typing import List


def generate_requests(
    n_requests: int,
    vocab_size: int = 32000,
    seq_len: int = 512,
    shared_prefix_len: int = 256,
    noncontiguous_ratio: float = 0.3,
    seed: int = 42,
) -> List[List[int]]:
    """Generate synthetic token sequences with controlled sharing patterns.

    Contiguous requests: shared_prefix + unique_suffix (standard prefix sharing).
    Non-contiguous requests: unique + shared_seg + unique + shared_seg
      — two shared segments inserted at chunk positions 1 and 3 so that
        each non-contiguous request can demonstrate two non-contiguous hits
        (one hit per shared segment, each preceded by a unique-chunk miss).
    """
    rng = random.Random(seed)
    chunk_size = seq_len // 4          # assume 4 chunks per request
    shared_prefix = [rng.randint(0, vocab_size - 1) for _ in range(shared_prefix_len)]
    shared_seg = shared_prefix[:chunk_size]  # first chunk of shared prefix

    requests: List[List[int]] = []
    for i in range(n_requests):
        if i < int(n_requests * (1 - noncontiguous_ratio)):
            # Contiguous prefix sharing: [shared(c0), shared(c1), unique(c2), unique(c3)]
            suffix = [rng.randint(0, vocab_size - 1) for _ in range(seq_len - shared_prefix_len)]
            tokens = shared_prefix[:shared_prefix_len] + suffix
        else:
            # Non-contiguous: [unique(c0), shared_seg(c1), unique(c2), shared_seg(c3)]
            # Two shared segments at positions 1 and 3 — each preceded by a unique miss
            unique_a = [rng.randint(0, vocab_size - 1) for _ in range(chunk_size)]
            unique_b = [rng.randint(0, vocab_size - 1) for _ in range(chunk_size)]
            tokens = unique_a + shared_seg + unique_b + shared_seg

        requests.append(tokens[:seq_len])
    return requests
