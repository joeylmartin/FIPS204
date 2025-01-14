from bitarray import bitarray
from bitarray.util import ba2int, int2ba
import numpy as np
from typing import Tuple
import math
from parametres import VECTOR_ARRAY_SIZE, Q_MODULUS, N_PRIVATE_KEY_RANGE, D_DROPPED_BITS, GAMMA_2_LOW_ORDER_ROUND, W_MAX_HINT_ONES

from bitarray import bitarray

def new_bitarray(*args, **kwargs):
    """
    Creates a bitarray with little-endian order by default.
    """
    kwargs['endian'] = 'little'
    return bitarray(*args, **kwargs)

def integer_to_bits(x: int, a: int) -> bitarray:
    '''
    Computes the base-2 representation of x mod 2a (using in little-endian order). 
    '''

    y = (x >> np.arange(a)) & 1
    
    # Initialize a bitarray and extend it with the computed bits
    bits = new_bitarray()
    bits.extend(y.astype(bool))  # Convert y to bool array, as bitarray works well with bools
    
    return bits
    
    #return int2ba(x[:a], endian="little")

def integer_to_bytes(x: int, a: int) -> bytes:
    '''
    Computes a base-256 representation of 𝑥 mod 256𝛼  using little-endian order.
    '''
    x_prime = x
    y = np.ndarray(a + 1, dtype='int')
    for i in range(a + 1):
        y[i] = x_prime % 256
        x_prime = x_prime // 256
    
    return y.tobytes()


#Bit to byte index conversion
bit_ind_conv = lambda bit_index : bit_index * 8
bit_arr_to_int = lambda bit_array : ba2int(bit_array)

def bits_to_integer(y: bitarray, a: int) -> int:
    '''
    Computes the integer represented by the base-2 representation y mod 2a.
    '''
    x = 0
    for i in range(1, a + 1):
        x = (2 * x) + y[a - i]
    return x
    '''
    if len(y) == 0:
        return 0
        
    return ba2int(y)
    '''

def coeff_from_three_bytes(b0: int, b1: int, b2: int) -> int:
    if b2 > 127:
        b2 -= 128
    z = (65536 * b2) + (256 * b1) + b0
    if z < Q_MODULUS:
        return z
    else:
        return None

def coeff_from_half_byte(b: int) -> int:
    '''
    Generates an element of {−η,−η +1,...,η} ∪ {None}. 
    '''
    if N_PRIVATE_KEY_RANGE == 2 and b < 15:
        return  2 - (b % 5)
    
    if N_PRIVATE_KEY_RANGE == 4 and b < 9:
        return 4 - b

    return None

def mod_pm(x, q):
    '''
    Computes x +-mod q, where BLAH TODO WRITE THIS.
    '''
    r = x % q
    if r > q // 2:
        r -= q
    return r

def power_2_round(r: int) -> Tuple[int, int]:
    '''
    Decomposes r into (r1, r0) such that r ≡ r1 2^d + r0 mod q.
    '''
    r_pos = r % Q_MODULUS
    r0 = mod_pm(r_pos, math.pow(2, D_DROPPED_BITS))
    return int((r_pos - r0) / (np.pow(2, D_DROPPED_BITS))), r0

def decompose(r: int) -> Tuple[int, int]:
    '''
    Decomposes r into (r1, r0) such that r ≡ r1(2*Gamma2) + r0 mod q.
    '''
    r_pos = r % Q_MODULUS
    r0 = mod_pm(r_pos, 2 * GAMMA_2_LOW_ORDER_ROUND)

    if r_pos - r0 == Q_MODULUS - 1:
        r1 = 0
        r0 -= 1
    else:
        r1 = (r_pos - r0) // (2 * GAMMA_2_LOW_ORDER_ROUND)
    return r1, r0

def high_bits(r: int) -> int:
    '''
    Retrieves the high bits from Decompose(r) output.
    '''
    r1, r0 = decompose(r)
    return r1

def low_bits(r: int) -> int:
    '''
    Retrieves the low bits from Decompose(r) output.
    '''
    r1, r0 = decompose(r)
    return r0

def make_hint(z: int, r: int)-> bool:
    '''
    Computes hint bit indicating whether adding 𝑧 to 𝑟 alters the high bits of 𝑟.
    '''
    r1 = high_bits(r)
    v1 = high_bits(r + z)
    return r1 != v1

#BOTTOM FOUR ORIGINALLY RETURN BYTE ARRAYS, WE'RE WORKING
#IN BIT ARRAYS

def simple_bit_pack(w: np.ndarray, b: int) -> bitarray:
    '''
    Encodes polynomial w into a bit string, such that the 
    coefficients of w are all in [0, b]
    ''' #BLAH FIX INT CASTING
    z = new_bitarray()
    for i in range(VECTOR_ARRAY_SIZE):
        z += integer_to_bits(int(w[i]), b.bit_length())
    return z

def bit_pack(w: np.ndarray, a: int, b: int) -> bitarray:
    '''
    Encodes a polynomial w into a bit string, such that the
    coefficients of w are all in [-a, b]
    '''
    z = new_bitarray()
    int_bl = (a + b).bit_length()
    for i in range(VECTOR_ARRAY_SIZE):
        z += integer_to_bits(int(b - w[i]), int_bl)
    return z

def simple_bit_unpack(v: bitarray, b: int) -> np.ndarray:
    '''
    Decodes a bit string v into a polynomial, such that the 
    coefficients of the polynomial are in [0, b]
    '''
    c = b.bit_length()
    w = np.empty(VECTOR_ARRAY_SIZE, dtype='int')
    for i in range(VECTOR_ARRAY_SIZE):
        w[i] = bits_to_integer(z[i * c : (i * c) + c - 1], c)
    return w

def bit_unpack(v: bitarray, a: int, b: int) -> np.ndarray:
    '''
    Decodes a bit string v into a polynomial, such that the 
    coefficients of the polynomial are in [-a, b]
    '''
    c = (a + b).bit_length()
    w = np.empty(VECTOR_ARRAY_SIZE, dtype='int')
    for i in range(VECTOR_ARRAY_SIZE):
        w[i] = b - bits_to_integer(v[i * c : (i * c) + c], c)
    return w

def hint_bit_pack(h: np.ndarray) -> bitarray:
    '''
    Encodes a hint vector h ^ k (of ring 2) into a bit string, such that the polynomial
    h's coefficients have at most w nonzeroes.
    '''
    y = new_bitarray()
    index = 0
    for i in range(K_MATRIX):
        for j in range(VECTOR_ARRAY_SIZE):
            if h[i][j] != 0:
                y[index] = j
                index += 1
        y[W_MAX_HINT_ONES + i] = index

    return z

def montgomery_reduce(a: int) -> int:
    '''
    Computes a * (2^-32) % q
    '''
    two32 = math.pow(2,32)
    QINV = 58728449 #inverse of q mod 2^32
    t = ((a % two32) * QINV) % two32
    r = (a - t * Q_MODULUS) // two32
    return r