"""Post-corrector: collapse stutter-style immediate word repetitions.

Conservative on purpose: only a small set of words are collapsed when
doubled. Doubling these is essentially always a disfluency, never grammatical
(unlike "had had"), so legitimate repeats are never destroyed.
"""

_COLLAPSIBLE = {"i", "the", "a", "an"}


def _bare(word: str) -> str:
    return word.lower().strip(".,!?;:'\"")


def correct(text: str) -> str:
    tokens = text.split()
    out: list[str] = []
    for token in tokens:
        if (out and _bare(token) == _bare(out[-1])
                and _bare(token) in _COLLAPSIBLE):
            continue
        out.append(token)
    return " ".join(out)
