"""Composes the post-processing correctors into one text transform.

Execution order is fixed (independent of the caller's enabled-list ordering):
numbers first (so spelling rules operate on words, not digit tokens), then
spelling_norm, then disfluency (so collapsing isn't fooled by spelling variants
of the same word).
"""

from postprocess import disfluency, numbers, spelling_norm

_ORDER = ["numbers", "spelling_norm", "disfluency"]


def make_pipeline(enabled):
    """Builds a `pipeline(text) -> str` applying the enabled correctors."""
    steps = []
    for name in _ORDER:
        if name not in enabled:
            continue
        if name == "numbers":
            steps.append(numbers.correct)
        elif name == "spelling_norm":
            steps.append(spelling_norm.correct)
        elif name == "disfluency":
            steps.append(disfluency.correct)

    def pipeline(text: str) -> str:
        for step in steps:
            text = step(text)
        return text

    return pipeline
