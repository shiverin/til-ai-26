from postprocess.pipeline import make_pipeline


def test_pipeline_applies_enabled_correctors_in_fixed_order():
    # numbers first ("3"->"three"), then spelling_norm (now identity — no
    # rules survived pruning), then disfluency ("the the"->"the").
    pipe = make_pipeline(["numbers", "spelling_norm", "disfluency"])
    assert pipe("the the centre 3") == "the centre three"


def test_empty_pipeline_is_identity():
    pipe = make_pipeline([])
    assert pipe("unchanged 3 centre text") == "unchanged 3 centre text"


def test_only_numbers_enabled():
    pipe = make_pipeline(["numbers"])
    assert pipe("the the centre 3") == "the the centre three"


def test_only_spelling_norm_enabled():
    # spelling_norm is now identity (all rules pruned — fired 0 clips on bench)
    pipe = make_pipeline(["spelling_norm"])
    assert pipe("the the centre 3") == "the the centre 3"


def test_only_disfluency_enabled():
    pipe = make_pipeline(["disfluency"])
    assert pipe("the the centre 3") == "the centre 3"


def test_order_is_fixed_regardless_of_enabled_list_order():
    # Even if caller passes them in reverse, the fixed order kicks in.
    pipe_a = make_pipeline(["disfluency", "spelling_norm", "numbers"])
    pipe_b = make_pipeline(["numbers", "spelling_norm", "disfluency"])
    assert pipe_a("the the centre 3") == pipe_b("the the centre 3")


def test_manual_corrections_enabled():
    # numbers -> spelling_norm -> manual_corrections -> disfluency
    pipe = make_pipeline(
        ["numbers", "spelling_norm", "manual_corrections", "disfluency"])
    # "sinite" -> "cyanite" (manual), "the the" -> "the" (disfluency),
    # "3" -> "three" (numbers).
    assert pipe("the the sinite 3 reserves") == "the cyanite three reserves"


def test_only_manual_corrections_enabled():
    pipe = make_pipeline(["manual_corrections"])
    assert pipe("the sinite is depleted") == "the cyanite is depleted"
