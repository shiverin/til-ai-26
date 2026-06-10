from postprocess.manual_corrections import correct, RULES


def test_substitutes_known_mishearing():
    assert (correct("the sinite reserves are depleted")
            == "the cyanite reserves are depleted")


def test_substitutes_possessive():
    assert correct("vika's report") == "devika's report"


def test_preserves_initial_capital():
    assert correct("Sinite operations") == "Cyanite operations"


def test_leaves_unmatched_words_alone():
    assert correct("the perimeter is clear") == "the perimeter is clear"


def test_all_rule_keys_lowercase():
    for key in RULES:
        assert key == key.lower(), key


def test_substitutes_apostrophe_word_boundary():
    # "castralian's" must match as a single token, not be cut at the apostrophe.
    assert correct("Castralian's fleet") == "Kestrelian's fleet"


def test_does_not_touch_substring():
    # 'wex' is a rule key; a longer word containing 'wex' must NOT be touched.
    assert correct("wexford station") == "wexford station"


def test_handles_all_ten_rules_present():
    expected = {"sinite", "firexis", "castralian", "castralian's", "vayanova",
                "vika's", "jahong", "zonone", "wex", "puel"}
    assert set(RULES.keys()) == expected
