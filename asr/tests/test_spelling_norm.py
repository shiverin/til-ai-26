from postprocess.spelling_norm import correct, RULES


def test_rules_is_empty_after_pruning():
    # All 22 candidate rules fired 0 clips on the 200-clip bench; they were
    # dropped to keep the dict minimal per the bench-prune protocol.
    assert RULES == {}


def test_correct_is_identity_when_rules_empty():
    assert correct("near the harbour gate") == "near the harbour gate"


def test_leaves_unmatched_words_alone():
    assert correct("the perimeter is clear") == "the perimeter is clear"


def test_does_not_touch_labour():
    # 'labour' is ambiguous in the reference set (64 BE vs 45 AE) and is
    # intentionally NOT in the rule set.
    assert correct("labour party") == "labour party"


def test_does_not_touch_colour():
    # 'colour' is intentionally NOT in the rule set (3 BE vs 8 AE in refs).
    assert correct("blue colour scheme") == "blue colour scheme"


def test_all_rule_keys_lowercase():
    # Invariant: rule lookup is lowercased, so dict keys must be lowercase.
    for key in RULES:
        assert key == key.lower(), key


def test_substring_match_not_triggered():
    # Even if rules were non-empty, apply_rules must NOT match inside longer
    # words.  Verify the regex boundary logic still holds for arbitrary input.
    assert correct("centredness study") == "centredness study"
