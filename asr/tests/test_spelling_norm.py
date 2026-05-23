from postprocess.spelling_norm import correct, RULES


def test_substitutes_be_to_ae_singular():
    assert correct("near the harbour gate") == "near the harbor gate"


def test_substitutes_plural():
    assert correct("two harbours") == "two harbors"


def test_preserves_initial_capital():
    assert correct("Defence forces") == "Defense forces"


def test_lowercase_input_stays_lowercase():
    assert correct("defence forces") == "defense forces"


def test_leaves_unmatched_words_alone():
    assert correct("the perimeter is clear") == "the perimeter is clear"


def test_does_not_touch_labour():
    # 'labour' is ambiguous in the reference set (64 BE vs 45 AE) and is
    # intentionally NOT in the rule set.
    assert correct("labour party") == "labour party"


def test_does_not_touch_colour():
    # 'colour' is intentionally NOT in the rule set (3 BE vs 8 AE in refs).
    assert correct("blue colour scheme") == "blue colour scheme"


def test_handles_apostrophe_word_boundary():
    assert correct("don't centre it") == "don't center it"


def test_all_rule_keys_lowercase():
    # Invariant: rule lookup is lowercased, so dict keys must be lowercase.
    for key in RULES:
        assert key == key.lower(), key


def test_substring_match_not_triggered():
    # 'centred' is a rule key — but it must NOT match inside a longer word.
    assert correct("centredness study") == "centredness study"
