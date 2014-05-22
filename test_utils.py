from nose.tools import assert_equal

import utils

__author__ = 'jyl111'


def test_gcd_correctness():
    assert_equal(utils.gcd(976, 1024), 16)
    assert_equal(utils.gcd(125, 1024), 1)
    assert_equal(utils.gcd(380, 1024), 4)
