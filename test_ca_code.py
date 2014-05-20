import ca_code
import numpy as np
from numpy.testing import assert_array_equal

first_tens_in_octal = [
    0o1440,  # PRN 1
    0o1620,  # PRN 2
    0o1710,  # PRN 3
    0o1744,  # PRN 4
    0o1133,  # PRN 5
    0o1455,  # PRN 6
    0o1131,  # PRN 7
    0o1454,  # PRN 8
    0o1626,  # PRN 9
    0o1504,  # PRN 10
    0o1642,  # PRN 11
    0o1750,  # PRN 12
    0o1764,  # PRN 13
    0o1772,  # PRN 14
    0o1775,  # PRN 15
    0o1776,  # PRN 16
    0o1156,  # PRN 17
    0o1467,  # PRN 18
    0o1633,  # PRN 19
    0o1715,  # PRN 20
    0o1746,  # PRN 21
    0o1763,  # PRN 22
    0o1063,  # PRN 23
    0o1706,  # PRN 24
    0o1743,  # PRN 25
    0o1761,  # PRN 26
    0o1770,  # PRN 27
    0o1774,  # PRN 28
    0o1127,  # PRN 29
    0o1453,  # PRN 30
    0o1625,  # PRN 31
    0o1712   # PRN 32
]

first_tens_in_binary_array = np.array([list(np.binary_repr(num)) for num in first_tens_in_octal], dtype=int)


def test_correctness():
    for i, expected in enumerate(first_tens_in_binary_array):
        yield check_correctness, i + 1, expected


def check_correctness(prn, expected):
    actual = ca_code.generate(prn=prn)
    assert_array_equal(actual[0, 0:10], expected)