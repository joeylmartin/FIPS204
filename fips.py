import random
import hashlib
import math
import pickle
from typing import Tuple
import numpy as np
from bitarray import bitarray, _set_default_endian
from parametres import *
from ntt_arithmetic import *

#d swap = DECLARATIVE IMPLEMENTATION OF IMPERATIVE STUFF


_set_default_endian(BYTEORDER)

#Bit to byte index conversion
bit_ind_conv = lambda bit_index : bit_index * 8
bit_arr_to_int = lambda bit_array : int(bit_array.to01(), 2)





def load_zeta_brv_cache(cache_file='zeta_brv_k_cache.pkl'):
    with open(cache_file, 'rb') as file:
        zeta_brv = pickle.load(file)
    return zeta_brv

# Example usage
cached_zeta_brv = load_zeta_brv_cache()



def NTT(w: np.ndarray) -> np.ndarray:
    w_hat = w.copy() #d swap
    
    k = 0
    len = 128
    while len >= 1:
        start = 0
        while start < 256:
            k += 1
            zeta = cached_zeta_brv[k]
            for j in range(start, start + len):
                t = (zeta * w_hat[j + len]) % Q_MODULUS
                w_hat[j + len] = (w_hat[j] - t) % Q_MODULUS
                w_hat[j] = (w_hat[j] + t) % Q_MODULUS
            start += 2 * len
        len = len // 2
    return w_hat

def NTT_inv(w_hat: np.ndarray) -> np.ndarray:
    w = w_hat.copy() #d swap
    
    k = 256
    len = 1
    while len < 256:
        start = 0
        while start < 256:
            k -= 1
            zeta = -cached_zeta_brv[k]
            for j in range(start, start + len):
                t = w[j]
                w[j] = (t + w[j + len]) % Q_MODULUS
                w[j + len] = (t - w[j + len]) % Q_MODULUS
                w[j+len] = (zeta * w[j+len]) % Q_MODULUS
            start += 2 * len
        len = 2 * len
    f = 8347681
    w = [(f * j) % Q_MODULUS for j in w] #d swap
    return w




def bitarr_get_byte(bits: bitarray, byte_index: int) -> int:
    slice0 = bit_ind_conv(byte_index)
    slice1 = bit_ind_conv(byte_index + 1)
    if(slice1 > len(bits)):
        raise ValueError(f"{byte_index} is out of range")
    
    return bit_arr_to_int(bits[slice0:slice1])




#USED FOR REJNTTPOLY AND EXPANDA


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

# USE A DECORATOR TO MANAGE EXTENDED SEED BETWEEN ROUNDS
def rej_ntt_poly(rho: bitarray):
    #where seed is bitstring of length 272

    j = 0
    c = 0
    a = np.empty(VECTOR_ARRAY_SIZE)


    #We extend the seed to source our coefficients (by indexing the C'th, C+1'th and C+2'th bytes). 
    #However, it is unbounded, and C will keep increasing until we have satifised 256 distinct coefficients.
    #We want to extend the  minimize the amount of times we have to extend the seed.

    extended_seed_length = bit_ind_conv(VECTOR_ARRAY_SIZE) * 3 #Minimum is 3 * 256 * 8= 768 bytes. TODO OPTIMIZE

    extended_seed = h_shake128(rho, extended_seed_length)

    while j < 256:

        #If we have exhausted the seed, extend it.
        if bit_ind_conv(c + 3) > extended_seed_length:
            extended_seed_length += VECTOR_ARRAY_SIZE
            #print(extended_seed_length)
            extended_seed = h_shake128(rho.tobytes(), extended_seed_length)
        
        byte_0 = bitarr_get_byte(extended_seed, c)        
        byte_1 = bitarr_get_byte(extended_seed, c + 1)             
        byte_2 = bitarr_get_byte(extended_seed, c + 2)     

        coeff = coeff_from_three_bytes(byte_0,byte_1, byte_2)
            
        c += 3

        if coeff is not None:
            a[j] = coeff
            j += 1


    return a

def rej_bounded_poly(rho: bitarray) -> np.ndarray:
    '''
    Samples an element a ∈ Rq with coeffcients in [−η,η] computed via rejection sampling from rho.
    Rho is a 528 bit string. 
    '''
    j = 0
    c = 0
    a = np.empty(VECTOR_ARRAY_SIZE)

    extended_seed_length = int(bit_ind_conv(VECTOR_ARRAY_SIZE) * 1.5 )#Minimum is 256 * 8= 2048 bytes. TODO OPTIMIZE

    extended_seed = h_shake128(rho.tobytes(),extended_seed_length)


    while j < VECTOR_ARRAY_SIZE:
        z = bitarr_get_byte(extended_seed, c)
        z0 = coeff_from_half_byte(z % 16)
        z1 = coeff_from_half_byte(z // 16)
        if z0 is not None:
            a[j] = z0
            j += 1
        if z1 is not None and j < VECTOR_ARRAY_SIZE:
            a[j] = z1
            j += 1
        c += 1

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
    matrix = np.empty((K_MATRIX, L_MATRIX, VECTOR_ARRAY_SIZE))

    for r in range(0, K_MATRIX):
        for s in range(0, L_MATRIX):
            matrix[r][s] = rej_ntt_poly(rho + integer_to_bits(s, 8) + integer_to_bits(r, 8))
    
    return matrix


def expand_s(rho: bitarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Samples vectors s1 ∈ Rlq and s2 ∈ Rkq, each with coeffcients in the interval [−η,η].
    '''
    s1, s2 = np.empty((L_MATRIX, VECTOR_ARRAY_SIZE), dtype='int'), np.empty((K_MATRIX, VECTOR_ARRAY_SIZE), dtype='int')
    for r in range(L_MATRIX):
        s1[r] = rej_bounded_poly(rho + integer_to_bits(r, 16))
    for r in range(K_MATRIX):
        s2[r] = rej_bounded_poly(rho + integer_to_bits(r + L_MATRIX, 16))
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

def ml_dsa_key_gen_internal(seed: bytes):

    #todo: update extension with matrix sampling!!! outdated
    #extend seed to 1024 bits with SHAKE-256
    extended_seed = h_shake256(seed, 1024)

    rho = extended_seed[:256] #first 256 bits
    rho_prime = extended_seed[256:768] #middle 512 bits
    k = extended_seed[768:] #last 256 bits

    a = expand_a(rho)
    s1, s2 = expand_s(rho_prime)
    ntt_expanded = [NTT(x) for x in s1]

    #product of A and NTT of S1
    ntt_product = matrix_vector_ntt(a, ntt_expanded)

    ntt_inv_expanded = [NTT_inv(x) for x in ntt_product]
    t = add_vector_ntt(ntt_inv_expanded, s2)


    #inner = np.matmul(a, ntt_temp)
    t = NTT_inv(inner) + s2

def ml_dsa_key_gen():
    '''
    Generates a public-private key pair. 
    '''
    #generate 256 bit seed
    seed = random.getrandbits(256) #change to approved RBG
    if seed == None:
        return None
    temp = seed.to_bytes(32, BYTEORDER)
    return ml_dsa_key_gen_internal(temp)

ml_dsa_key_gen()