"""Versioned synthetic generators. Default legacy behavior remains unchanged.

Opt into profile="sequences" for failure deliveries with point-in-time features.
Use generate_dataset() for successes, outcomes and infrastructure episode truth.
"""
from eval.legacy_generate import (
    CAUSE_PRIOR, CODE_EMISSION, METHOD_BY_CAUSE, NOISE, RAILS,
    assign_arm, split_hash, generate as generate_legacy,
)


def generate(n=5000, seed=42, *, profile="legacy", include_candidates=False):
    if profile == "legacy":
        return generate_legacy(n=n, seed=seed)
    if profile == "sequences":
        return generate_dataset(n=n, seed=seed, include_candidates=include_candidates).deliveries
    raise ValueError(f"Unknown generator profile: {profile}")


def generate_dataset(n=20000, seed=42, *, include_candidates=False):
    from eval.sequences import generate_dataset as sequences
    return sequences(n=n, seed=seed, include_candidates=include_candidates)
