"""Post-corrector: hand-curated mishearing fixes.

Each rule is a wrong->right substitution Parakeet emits as a mishearing of a
proper noun (or nonsense word) in this corpus. All rules are blanket-safe:
the "wrong" word does NOT appear in any of the 4110 novice reference
transcripts, so the substitution cannot break a clip where the model was
right.

Discovered by word-level diffing prediction vs reference on the 200-clip
asr_shizhen bench, then filtered to (hyp, ref) pairs where the hyp word is
absent from the full reference vocabulary.
"""

from postprocess.spelling_norm import apply_rules

RULES: dict[str, str] = {
    "sinite": "cyanite",
    "firexis": "phyrexis",
    "castralian": "kestrelian",
    "castralian's": "kestrelian's",
    "vayanova": "veyanova",
    "vika's": "devika's",
    "jahong": "jiahong",
    "zonone": "zonnon",
    "wex": "vex",
    "puel": "fuel",
}


def correct(text: str) -> str:
    """Apply the manual-correction rule set."""
    return apply_rules(text, RULES)
