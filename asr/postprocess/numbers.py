"""Post-corrector: spell out digit runs so they match the reference style.

The ASR model sometimes emits digits ("190", "3") while the reference
transcripts always spell numbers out. This corrector converts digit runs to
words: a 1-2 digit run with no leading zero becomes cardinal words
("48" -> "forty eight"); anything longer, or with a leading zero, is spelled
digit-by-digit ("190" -> "one nine zero"); a decimal point becomes "point".
"""

import re

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine"]
_TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
          "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]

_DIGIT_RUN = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?(?![A-Za-z])")


def _cardinal(n: int) -> list[str]:
    if n < 10:
        return [_ONES[n]]
    if n < 20:
        return [_TEENS[n - 10]]
    tens, ones = divmod(n, 10)
    return [_TENS[tens]] if ones == 0 else [_TENS[tens], _ONES[ones]]


def _digit_by_digit(digits: str) -> list[str]:
    return [_ONES[int(d)] for d in digits]


def _convert_run(digits: str) -> list[str]:
    if len(digits) <= 2 and not digits.startswith("0"):
        return _cardinal(int(digits))
    return _digit_by_digit(digits)


def correct(text: str) -> str:
    def repl(match: re.Match) -> str:
        token = match.group(0)
        if "." in token:
            left, right = token.split(".", 1)
            words = _convert_run(left) + ["point"] + _digit_by_digit(right)
        else:
            words = _convert_run(token)
        return " ".join(words)

    return _DIGIT_RUN.sub(repl, text)
