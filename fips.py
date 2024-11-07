import random
import hashlib
import math
from typing import Tuple
import numpy as np
from bitarray import bitarray, _set_default_endian




BYTEORDER='little' #CHECK IF ENDIANNESS MESSES STUFF UP!!

_set_default_endian(BYTEORDER)

#Bit to byte index conversion
bit_ind_conv = lambda bit_index : bit_index * 8
bit_arr_to_int = lambda bit_array : int(bit_array.to01(), 2)


#PARAMETRES e.g

STRENGTH = "ML-DSA-44" #ML-DSA-65, MLA-DSA-87

#match STRENGTH:
#    case "ML-DSA-44":
Q_MODULUS = 8380417 
D_DROPPED_BITS = 13 
TAU_ONES = 39
LAMBDA_COLLISION_STR = 128
GAMMA_1_COEFFICIENT = math.pow(2, 17)
GAMMA_2_LOW_ORDER_ROUND = (Q_MODULUS - 1)/88
K_MATRIX = 4
L_MATRIX = 4
N_PRIVATE_KEY_RANGE = 2
THETA_ = TAU_ONES * N_PRIVATE_KEY_RANGE
W_MAX_HINT_ONES = 80
    #case _:


#USED FOR REJNTTPOLY AND EXPANDA
COEFFICIENT_ARRAY_SIZE = 256

def coeff_from_three_bytes(b0: int, b1: int, b2: int) -> int:
    if b2 > 127:
        b2 -= 128
    z = (65536 * b2) + (256 * b1) + b0
    if z < Q_MODULUS:
        return z
    else:
        return None

# USE A DECORATOR TO MANAGE EXTENDED SEED BETWEEN ROUNDS
def rej_ntt_poly(seed: bitarray):
    #where seed is bitstring of length 272

    j = 0
    c = 0
    a = np.empty(COEFFICIENT_ARRAY_SIZE)


    #We extend the seed to source our coefficients (by indexing the C'th, C+1'th and C+2'th bytes). 
    #However, it is unbounded, and C will keep increasing until we have satifised 256 distinct coefficients.
    #We want to extend the  minimize the amount of times we have to extend the seed.

    extended_seed_length = COEFFICIENT_ARRAY_SIZE * 10 #Minimum is 3 * 256 = 768 bits. * 4 to be safe; CAN OPTIMIZE THIS.

    extended_seed = h_shake128(seed, extended_seed_length)

    while j < 256:

        #If we have exhausted the seed, extend it.
        if bit_ind_conv(c + 3) > extended_seed_length:
            extended_seed_length += COEFFICIENT_ARRAY_SIZE
            #print(extended_seed_length)
            extended_seed = h_shake128(seed.tobytes(), extended_seed_length)
        
        byte_0 = extended_seed[bit_ind_conv(c): bit_ind_conv(c+1)]        
        byte_1 = extended_seed[bit_ind_conv(c+1): bit_ind_conv(c+2)]
        byte_2 = extended_seed[bit_ind_conv(c+2): bit_ind_conv(c+3)]

        coeff = coeff_from_three_bytes(bit_arr_to_int(byte_0),bit_arr_to_int(byte_1), bit_arr_to_int(byte_2))
            
        c += 3

        if coeff is not None:
            a[j] = coeff
            j += 1
    print(c)

    return a

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


def expand_a(rho: bitarray) -> np.ndarray:
    '''
    Samples a k x l matrix A of elements of Tq; where Rho is a 256 bit string
    '''
    matrix = np.empty((K_MATRIX, L_MATRIX, COEFFICIENT_ARRAY_SIZE))

    for r in range(0, K_MATRIX):
        for s in range(0, L_MATRIX):
            matrix[r][s] = rej_ntt_poly(rho + integer_to_bits(s, 8) + integer_to_bits(r, 8))
    
    return matrix


def expand_s(rho_prime: bitarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Samples vectors s1 ∈ Rlq and s2 ∈ Rkq, each with coeffcients in the interval [−η,η].
    '''
    s1, s2 = np.empty(L_MATRIX), np.empty(K_MATRIX)
    for r in range(L_MATRIX):
        s1[r] = 2 #rejbounded poly
    for r in range(K_MATRIX):
        s2[r] = 2
    return s1, s2

def h_shake256(seed: bytes, bit_length: int) -> bitarray:
    '''
    Extend bit string seed with SHAKE-256 XOF
    '''
    byte_length = bit_length // 8

    hash_obj = hashlib.shake_256()
    hash_obj.update(seed)
    xof_output = hash_obj.digest(byte_length)

    bits = bitarray()
    #Return bitarray object from bytes
    bits.frombytes(xof_output)
    return bits

def h_shake128(seed: bytes, bit_length: int) -> bitarray:
    '''
    Extend bit string seed with SHAKE-128 XOF
    '''
    byte_length = bit_length // 8

    hash_obj = hashlib.shake_128()
    hash_obj.update(seed)
    xof_output = hash_obj.digest(byte_length)

    #Return bitarray object from bytes
    bits = bitarray()
    #Return bitarray object from bytes
    bits.frombytes(xof_output)
    return bits

def ml_dsa_key_gen():
    '''
    Generates a public-private key pair. 
    '''
    #generate 256 bit seed
    seed = random.getrandbits(256) #change to approved RBG
    bin_seed = format(seed, '0b') #conv to binary

    #extend seed to 1024 bits with SHAKE-256
    extended_seed = h_shake256(bin_seed.encode(), 1024)

    rho = extended_seed[:256] #first 256 bits
    rho_prime = extended_seed[256:768] #middle 512 bits
    k = extended_seed[768:] #last 256 bits

    a = expand_a(rho)
    print('g')


ml_dsa_key_gen()