import random
import hashlib
import math
import pickle
from typing import Tuple
import numpy as np
from bitarray import bitarray, _set_default_endian
from parametres import *
from ntt_arithmetic import *
from auxiliary_funcs import *
from hash_funcs import h_shake256, h_shake128

#d swap = DECLARATIVE IMPLEMENTATION OF IMPERATIVE STUFF


_set_default_endian(BYTEORDER)

#Bit to byte index conversion
bit_ind_conv = lambda bit_index : bit_index * 8
bit_arr_to_int = lambda bit_array : int(bit_array.to01(), 2)

def mod_pm(x, q):
    r = x % q
    if r > q // 2:
        r -= q
    return r



def load_zeta_brv_cache(cache_file='zeta_brv_k_cache.pkl'):
    with open(cache_file, 'rb') as file:
        zeta_brv = pickle.load(file)
    return zeta_brv

# Example usage
cached_zeta_brv = load_zeta_brv_cache()


def power_2_round(r) -> Tuple[int, int]:
    '''
    Decomposes r into (r1, r0) such that r ≡ r1 2^d + r0 mod q.
    '''
    r_pos = r % Q_MODULUS
    r_0 = mod_pm(r_pos, np.pow(2, D_DROPPED_BITS))
    return (r_pos - r_0) / (np.pow(2, D_DROPPED_BITS)), r_0

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
    
    m = 256
    len = 1
    while len < 256:
        start = 0
        while start < 256:
            m -= 1
            zeta = -cached_zeta_brv[m]
            for j in range(start, start + len):
                t = w[j]
                w[j] = (t + w[j + len]) % Q_MODULUS
                w[j + len] = (t - w[j + len]) % Q_MODULUS
                w[j+len] = (zeta * w[j+len]) % Q_MODULUS
            start += 2 * len
        len *= 2
    f = 8347681
    w = [(f * j) % Q_MODULUS for j in w] #d swap
    return w




def bitarr_get_byte(bits: bitarray, byte_index: int) -> int:
    slice0 = bit_ind_conv(byte_index)
    slice1 = bit_ind_conv(byte_index + 1)
    if(slice1 > len(bits)):
        raise ValueError(f"{byte_index} is out of range")
    
    return bit_arr_to_int(bits[slice0:slice1])


#ORIGINALLY BYTE ARRAY
def pkEncode(rho : bitarray, t1 : np.ndarray) -> bitarray:
    '''
    Encodes a public key for ML-DSA into a byte string.
    Where T1 is K polynomials (k x 256 np array)
    '''
    pk = rho.copy()
    for i in range(K_MATRIX):
        pk = pk + simple_bit_pack(t1[i], 10)
    return pk

def skEncode(rho : bitarray, K : int, tr : bitarray, s1 : np.ndarray, s2 : np.ndarray, t0 : np.ndarray) -> bitarray:
    sk = rho + integer_to_bits(K) + tr
    for i in range(L_MATRIX):
        sk += bit_pack(s1[i], N_PRIVATE_KEY_RANGE, N_PRIVATE_KEY_RANGE)
    
    for i in range(K_MATRIX):
        sk += bit_pack(s2[i], N_PRIVATE_KEY_RANGE, N_PRIVATE_KEY_RANGE)
    
    lower_bound = math.pow(2, D_DROPPED_BITS - 1) - 1
    for i in range(K_MATRIX):
        sk += bit_pack(t0[i], lower_bound, lower_bound + 1)
    
    return sk

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



def ml_dsa_key_gen_internal(seed: bytes):

    #todo: update extension with matrix sampling!!! outdated
    #extend seed to 1024 bits with SHAKE-256
    extended_seed = h_shake256(seed, 1024)

    rho = extended_seed[:256] #first 256 bits
    rho_prime = extended_seed[256:768] #middle 512 bits
    k = extended_seed[768:] #last 256 bits

    #Create Matrix A, and vectors s1 (secret) and s2 (error)
    a = expand_a(rho)
    s1, s2 = expand_s(rho_prime)

    #product of A and NTT of S1
    ntt_product = matrix_vector_ntt(a, [NTT(x) for x in s1])
    t = add_vector_ntt([NTT_inv(x) for x in ntt_product], s2)


    #TODO: double check which uses L_MATRIX!! One of them is wrong, maybe?
    t0 = np.ndarray((K_MATRIX, VECTOR_ARRAY_SIZE))
    t1 = np.ndarray((K_MATRIX, VECTOR_ARRAY_SIZE))

    #apply power2round component-wise
    for i in range(K_MATRIX):
        for j in range(VECTOR_ARRAY_SIZE):
            t1[i][j], t0[i][j] = power_2_round(t[i][j])

    pk = pkEncode(rho, t1)
    tr = h_shake256(pk.tobytes(), 64)
    sk = skEncode(rho, K_MATRIX, s1, s2, t0)
    return pk, sk

def ml_dsa_key_gen():
    '''
    Generates a public-private key pair. 
    '''
    #generate 256 bit seed
    seed = random.getrandbits(256) #change to approved RBG

    if seed == None: #needed?
        return None

    temp = seed.to_bytes(32, BYTEORDER)
    return ml_dsa_key_gen_internal(temp)

ml_dsa_key_gen()