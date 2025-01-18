from bitarray import bitarray
from bitarray.util import ba2int, int2ba
import numpy as np
from typing import Tuple
import math
from parametres import BYTEORDER, K_MATRIX, VECTOR_ARRAY_SIZE, Q_MODULUS, N_PRIVATE_KEY_RANGE, D_DROPPED_BITS, GAMMA_2_LOW_ORDER_ROUND, W_MAX_HINT_ONES

from bitarray import bitarray

def new_bitarray(*args, **kwargs):
    """
    Creates a bitarray with little-endian order by default.
    """
    kwargs['endian'] = 'little'
    return bitarray(*args, **kwargs)

def bitarr_get_byte(bits: bitarray, byte_index: int) -> int:
    '''
    Gets the byte at byte_index in a bitarray.
    '''
    slice0 = bit_ind_conv(byte_index)
    slice1 = bit_ind_conv(byte_index + 1)
    if(slice1 > len(bits)):
        raise ValueError(f"{byte_index} is out of range")
    
    return bit_arr_to_int(bits[slice0:slice1])

def bitarr_set_byte(bits: bitarray, byte_index: int, value: int):
    '''
    Sets the byte at byte_index in a bitarray to value in bits.
    '''
    slice0 = bit_ind_conv(byte_index)
    slice1 = bit_ind_conv(byte_index + 1)
    bits[slice0:slice1] = int2ba(value, 8)

def integer_to_bits(x: int, a: int) -> bitarray:
    '''
    Computes the base-2 representation of x mod 2a (using in little-endian order). 
    '''
    x_h = x
    y = list()
    for i in range(a):
        y.append(x_h % 2)
        x_h = x_h // 2
    return new_bitarray(y)
    '''
    y = (x >> np.arange(a)) & 1
    
    # Initialize a bitarray and extend it with the computed bits
    bits = new_bitarray()
    bits.extend(y.astype(bool))  # Convert y to bool array, as bitarray works well with bools
    
    return bits'''

    #return int2ba(x[:a], endian="little")

def integer_to_bytes(x: int, a: int) -> bytes:
    '''
    Computes a base-256 representation of 𝑥 mod 256𝛼  using little-endian order.
    '''
    x_prime = x
    y = np.ndarray(a, dtype='uint8')
    for i in range(a):
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
    b2_p = b2
    if b2_p > 127:
        b2_p -= 128

    z = (65536 * b2_p) + (256 * b1) + b0
    if z < Q_MODULUS:
        return z
    else:
        return None

def coeff_from_half_byte(b: int) -> int:
    '''
    Generates an element of {−η,−η +1,...,η} ∪ {None}. 
    '''
    if N_PRIVATE_KEY_RANGE == 2 and b < 15:
        return 2 - (b % 5)
    
    if N_PRIVATE_KEY_RANGE == 4 and b < 9:
        return 4 - b

    return None

def mod_pm(x, q):
    '''
    Computes x +-mod q, under symmetric modular arithmetic,
    where -(Q/2) < x < (Q/2).
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
    two_d = math.pow(2, D_DROPPED_BITS)
    r0 = mod_pm(r_pos, two_d)
    return (r_pos - r0) // two_d, r0

def decompose(r: int) -> Tuple[int, int]:
    '''
    Decomposes r into (r1, r0) such that r ≡ r1(2*Gamma2) + r0 mod q.
    '''
    r_pos = r % Q_MODULUS
    r0 = mod_pm(r_pos, 2 * GAMMA_2_LOW_ORDER_ROUND)

    if r_pos - r0 + 1 == Q_MODULUS:
        r1 = 0
        r0 -= 1
    else:
        r1 = (r_pos - r0) // (2 * GAMMA_2_LOW_ORDER_ROUND)
    return r1, r0

def high_bits(r: int) -> int:
    '''
    Retrieves the high bits from Decompose(r) output.
    '''
    r1, _ = decompose(r)
    return r1

def low_bits(r: int) -> int:
    '''
    Retrieves the low bits from Decompose(r) output.
    '''
    _, r0 = decompose(r)
    return r0

def make_hint(z: int, r: int)-> bool:
    '''
    Computes hint bit indicating whether adding 𝑧 to 𝑟 alters the high bits of 𝑟.
    '''
    r1 = high_bits(r)
    v1 = high_bits(r + z)
    return r1 != v1

def use_hint(h: bool, r: int) -> int:
    '''
    Returns the high bits of 𝑟 adjusted according to hint ℎ.
    '''
    m = (Q_MODULUS - 1)/(2 * GAMMA_2_LOW_ORDER_ROUND)
    r1, r0 = decompose(r)

    if h and r0 > 0:
        return (r1 + 1) % m
    
    if h and r0 <= 0:
        return (r1 - 1) % m
    
    return r1

#BOTTOM FOUR ORIGINALLY RETURN BYTE ARRAYS, WE'RE WORKING
#IN BIT ARRAYS

def simple_bit_pack(w: np.ndarray, b: int) -> bitarray:
    '''
    Encodes polynomial w into a bit string, such that the 
    coefficients of w are all in [0, b]
    ''' #BLAH FIX INT CASTING
    z = new_bitarray()
    for i in range(VECTOR_ARRAY_SIZE):
        z += integer_to_bits(w[i], b.bit_length())
    return z

def simple_bit_unpack(v: bitarray, b: int) -> np.ndarray:
    '''
    Decodes a bit string v into a polynomial, such that the 
    coefficients of the polynomial are in [0, b]
    '''
    c = b.bit_length()
    w = np.empty(VECTOR_ARRAY_SIZE, dtype='int64')
    for i in range(VECTOR_ARRAY_SIZE):
        w[i] = bits_to_integer(v[i * c : (i * c) + c], c)
    return w

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

def bit_unpack(v: bitarray, a: int, b: int) -> np.ndarray:
    '''
    Decodes a bit string v into a polynomial, such that the 
    coefficients of the polynomial are in [-a, b]
    '''
    c = (a + b).bit_length()
    w = np.empty(VECTOR_ARRAY_SIZE, dtype='int64')
    for i in range(VECTOR_ARRAY_SIZE):
        w[i] = b - bits_to_integer(v[i * c : (i * c) + c], c)
    return w

def hint_bit_pack(h: np.ndarray) -> bitarray:
    '''
    Encodes a hint vector h ^ k (of ring 2) into a bit string, such that the polynomial
    h's coefficients have at most w nonzeroes. Returns bitarray
    '''

    ###Create bitarray y, initialized to (W + K) * 8 bits
    init_0s = (0).to_bytes(K_MATRIX + W_MAX_HINT_ONES, BYTEORDER)
    y = new_bitarray()
    y.frombytes(init_0s)

    index = 0
    for i in range(K_MATRIX):
        for j in range(VECTOR_ARRAY_SIZE):
            if h[i][j] != 0:
                bitarr_set_byte(y, index, j)
                index += 1
        bitarr_set_byte(y, W_MAX_HINT_ONES + i, index)

    return y

def hint_bit_unpack(y: bitarray) -> np.ndarray: 
    h = np.zeros((K_MATRIX, VECTOR_ARRAY_SIZE), dtype='int64')
    index = 0
    for i in range(K_MATRIX):
        wi_byte = bitarr_get_byte(y, W_MAX_HINT_ONES + i)
        if wi_byte < index or wi_byte > W_MAX_HINT_ONES:
            return None #Malformed hint
        
        first = index
        while index < wi_byte:
            if index > first:
                if bitarr_get_byte(y, index -1) >= bitarr_get_byte(y, index):
                    return None #Malformed input
            
            h[i][bitarr_get_byte(y, index)] = 1
            index += 1
    
    for i in range(index, W_MAX_HINT_ONES): #read leftover bytes
        if bitarr_get_byte(y, i) != 0:
            return None #Malformed hint
    return h
        

def montgomery_reduce(a: int) -> int:
    '''
    Computes a * (2^-32) % q
    '''
    two32 = math.pow(2,32)
    QINV = 58728449 #inverse of q mod 2^32
    t = ((a % two32) * QINV) % two32
    r = (a - t * Q_MODULUS) // two32
    return r

def get_vector_infinity_norm(vector: np.ndarray) -> int:
    '''
    Gets the infinity norm of a vector, which uses symmetric modulo representation
    (-Q_MODULUS/2 < inf_norm < Q_MODULUS/2) 
    '''
    max_val = vector.max()
    return mod_pm(max_val, Q_MODULUS)