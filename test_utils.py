from nose.tools import assert_equal

import utils

__author__ = 'jyl111'


def test_gcd_correctness():
    assert_equal(utils.gcd(976, 1024), 16)
    assert_equal(utils.gcd(125, 1024), 1)
    assert_equal(utils.gcd(380, 1024), 4)

def test_mod_inverse_correctness():
    assert_equal(utils.mod_inverse(a=3, n=37), 25)
    assert_equal(utils.mod_inverse(a=7, n=26), 15)
    assert_equal(utils.mod_inverse(a=1, n=10), 1)
