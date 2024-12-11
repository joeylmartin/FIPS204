from bitarray import bitarray
import numpy as np
from parametres import VECTOR_ARRAY_SIZE, Q_MODULUS, N_PRIVATE_KEY_RANGE

def integer_to_bits(x: int, a: int) -> bitarray:
    '''
    Computes the base-2 representation of x mod 2a (using in little-endian order). 
    '''

    #TO DO: MAKE THIS SIMPLER!!!

    y = (x >> np.arange(a)) & 1
    
    # Initialize a bitarray and extend it with the computed bits
    bits = bitarray()
    bits.extend(y.astype(bool))  # Convert y to bool array, as bitarray works well with bools
    
    return bits


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


#BOTTOM TWO ORIGINALLY RETURN BYTE ARRAYS, WE'RE WORKING
#IN BIT ARRAYS

def simple_bit_pack(w: np.ndarray, b: int) -> bitarray:
    '''
    Encodes polynomial w into a bit string, such that the 
    coefficients of w are all in [0, b]
    '''
    z = bitarray()
    for i in range(VECTOR_ARRAY_SIZE):
        z += integer_to_bits(w[i], b.bit_length())
    return z

def bit_pack(w: np.ndarray, a: int, b: int) -> bitarray:
    '''
    Encodes a polynomial w into a bit string, such that the
    coefficients of w are all in [-a, b]
    '''
    z = bitarray()
    for i in range(VECTOR_ARRAY_SIZE):
        z += integer_to_bits(b - w[i], (a + b).bit_length())
    return z
