from __future__ import division

import fractions

import numpy as np


__author__ = 'jyl111'


gcd = np.frompyfunc(fractions.gcd, 2, 1)


def extended_gcd(a, b):
    """
    Extended GCD using Extended Euclidean algorithm
    References:
        http://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
        https://en.wikibooks.org/wiki/Algorithm_Implementation/Mathematics/Extended_Euclidean_algorithm
    """
    if a == 0:
        return b, 0, 1
    else:
        g, y, x = extended_gcd(b % a, a)
        return g, x - (b // a) * y, y


def mod_inverse(a, n):
    """
    Modular inversion using the extended Euclidean algorithm
    References:
        http://rosettacode.org/wiki/Modular_inverse#Python
        https://en.wikibooks.org/wiki/Algorithm_Implementation/Mathematics/Extended_Euclidean_algorithm#Modular_inverse
    """
    gcd, x, y = extended_gcd(a, n)
    if gcd != 1:
        return None # modular inverse does not exist
    else:
        return x % n

    # t = 0
    # new_t = 1
    # r = n
    # new_r = a
    #
    # while new_r > 0:
    #     quotient = r % new_r
    #     t, new_t = new_t, (t - quotient * new_t)
    #     r, new_r = new_r, (r - quotient * new_r)
    #     print quotient, t, r
    #
    # if r > 1:
    #     raise ValueError('a is not invertible')
    #
    # if t < 0:
    #     t += n
    #
    # return t


def floor_to_pow2(a):
    ans = 1

    while ans <= a:
        ans <<= 1

    return int(ans/2)
