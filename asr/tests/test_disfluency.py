from postprocess.disfluency import correct


def test_collapses_doubled_pronoun():
    assert correct("yeah I I know") == "yeah I know"


def test_collapses_doubled_article():
    assert correct("the the signal") == "the signal"


def test_does_not_collapse_legitimate_doubling():
    # "had had" is grammatical (past perfect) -> must be left alone.
    assert correct("he had had enough") == "he had had enough"


def test_no_repeats_unchanged():
    assert correct("no repeats in this line") == "no repeats in this line"
