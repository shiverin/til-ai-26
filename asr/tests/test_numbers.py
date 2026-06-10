from postprocess.numbers import correct


def test_single_digit_becomes_cardinal():
    assert correct("checkpoint sentinel 3") == "checkpoint sentinel three"


def test_two_digit_becomes_cardinal():
    assert correct("past 48 hours") == "past forty eight hours"


def test_three_digit_run_is_digit_by_digit():
    assert correct("bearing 190 degrees") == "bearing one nine zero degrees"


def test_decimal_uses_point():
    assert correct("2.3 million phi") == "two point three million phi"


def test_leading_zero_is_digit_by_digit():
    assert correct("code 07") == "code zero seven"


def test_text_without_digits_is_unchanged():
    assert correct("no digits in this sentence") == "no digits in this sentence"


def test_digit_glued_to_letters_is_left_alone():
    assert correct("phi3 model alpha7") == "phi3 model alpha7"
