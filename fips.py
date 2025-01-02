import random
import hashlib
import math
import pickle
from typing import Tuple, Any
import numpy as np
from bitarray import bitarray, _set_default_endian
from parametres import *
from ntt_arithmetic import *
from auxiliary_funcs import *
from hash_funcs import h_shake256, h_shake128, h_init, hash_absorb, hash_squeeze

#d swap = DECLARATIVE IMPLEMENTATION OF IMPERATIVE STUFF

#TODO: verify dtype of np arrays!!

#byteorder is little endian
_set_default_endian(BYTEORDER)


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

def skEncode(rho : bitarray, k : bitarray, tr : bitarray, s1 : np.ndarray, s2 : np.ndarray, t0 : np.ndarray) -> bitarray:
    sk = rho + k + tr
    for i in range(L_MATRIX):
        sk += bit_pack(s1[i], N_PRIVATE_KEY_RANGE, N_PRIVATE_KEY_RANGE)
    
    for i in range(K_MATRIX):
        sk += bit_pack(s2[i], N_PRIVATE_KEY_RANGE, N_PRIVATE_KEY_RANGE)
    
    lower_bound = int(math.pow(2, D_DROPPED_BITS - 1) - 1)
    for i in range(K_MATRIX):
        sk += bit_pack(t0[i], lower_bound, lower_bound + 1)
    
    return sk

def skDecode(sk : bitarray) -> Tuple[bitarray, bitarray, bitarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Reverses the procedure skEncode, where SK is a 
    32 + 32 + 64 + 32 x (bitlen(l+k) + (2n) + dk + 8) bit string
    '''
    rho = sk[:256]
    k = sk[256:512]
    tr = sk[512:1024]

    y = np.ndarray(L_MATRIX, dtype=bitarray)
    z = np.ndarray(K_MATRIX, dtype=bitarray)
    w = np.ndarray(K_MATRIX, dtype=bitarray)

    #TODO: figure out appropriate var title for this! What are Y and Z?
    y_z_len = 8 * 32 * (2 * N_PRIVATE_KEY_RANGE).bit_length()


    #Retrieve Y, Z, and W arrays from SK
    index = 1024

    for i in range(L_MATRIX):
        y[i] = sk[index : index + y_z_len]
        index += y_z_len

    for i in range(K_MATRIX):
        z[i] = sk[index : index + y_z_len]
        index += y_z_len

    w_len = D_DROPPED_BITS * 8 * 32
    for i in range(K_MATRIX):
        w[i] = sk[index : index + w_len]
        index += w_len

    s1 = np.ndarray((L_MATRIX,VECTOR_ARRAY_SIZE), dtype='int')
    s2 = np.ndarray((K_MATRIX,VECTOR_ARRAY_SIZE), dtype='int')
    t0 = np.ndarray((K_MATRIX,VECTOR_ARRAY_SIZE), dtype='int')

    for i in range(L_MATRIX):
        s1[i] = bit_unpack(y[i], N_PRIVATE_KEY_RANGE, N_PRIVATE_KEY_RANGE)
    for i in range(K_MATRIX):
        s2[i] = bit_unpack(z[i], N_PRIVATE_KEY_RANGE, N_PRIVATE_KEY_RANGE)
    
    lower_bound = int(math.pow(2, D_DROPPED_BITS - 1) - 1)
    for i in range(K_MATRIX):
        t0[i] = bit_unpack(w[i], lower_bound, lower_bound + 1)
    
    return rho, k, tr, s1, s2, t0


def w1_encode(w1: np.ndarray) -> bitarray:
    '''
    Encodes a polynomial w1 into a bit string.
    '''
    w_hat = new_bitarray()
    bit_pack_bound= int((Q_MODULUS - 1)/(2 * GAMMA_2_LOW_ORDER_ROUND) - 1)
    for i in range(K_MATRIX):
        w_hat += simple_bit_pack(w1[i], bit_pack_bound)
    return w_hat




#PSEUDO-RANDOM SAMPLING FUNCTIONS

def sample_in_ball(rho: bitarray) -> np.ndarray:

    c = np.zeros(VECTOR_ARRAY_SIZE, dtype='int')

    ctx = h_init()
    ctx = hash_absorb(ctx, rho)
    ctx, h = hash_squeeze(ctx, 64)

    for i in range(256 - TAU_ONES, 256):
        j = 256

        while j > i:
            ctx, j_b = hash_squeeze(ctx, 8)
            j = bit_arr_to_int(j_b)

        c[i] = c[j]
        temp = h[i+TAU_ONES - 256]
        c[j] = (-1) ** temp

    return c
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
    Samples vectors s1 ∈ Rlq and s2 ∈ Rkq, each with 
    coeffcients in the interval [−η,η].
    '''
    s1, s2 = np.empty((L_MATRIX, VECTOR_ARRAY_SIZE), dtype='int'), np.empty((K_MATRIX, VECTOR_ARRAY_SIZE), dtype='int')
    for r in range(L_MATRIX):
        s1[r] = rej_bounded_poly(rho + integer_to_bits(r, 16))
    for r in range(K_MATRIX):
        s2[r] = rej_bounded_poly(rho + integer_to_bits(r + L_MATRIX, 16))
    return s1, s2

def expand_mask(rho: bitarray, mu: int) -> np.ndarray:
    '''
    Samples a vector y ∈ Rl  such that each polynomial y[r] 
    has coefficients between −𝛾 +1 and 𝛾1. 
    '''
    y = np.empty((L_MATRIX,VECTOR_ARRAY_SIZE), dtype='int')

    c = 1 + (GAMMA_1_COEFFICIENT - 1).bit_length()

    for r in range(L_MATRIX):
        rho_prime = rho + integer_to_bits(mu + r, 16)
        v = h_shake256(rho_prime.tobytes(), 32 * 8 * c)
        y[r] = bit_unpack(v, GAMMA_1_COEFFICIENT-1, GAMMA_1_COEFFICIENT)
    
    return y

def ml_dsa_key_gen_internal(seed: bytes) -> Tuple[bitarray, bitarray]:

    #todo: update extension with matrix sampling!!! outdated
    #extend seed to 1024 bits with SHAKE-256

    h_input = seed + integer_to_bytes(K_MATRIX, 1) + integer_to_bytes(L_MATRIX, 1)

    extended_seed = h_shake256(h_input, 1024)

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
    t0 = np.ndarray((K_MATRIX, VECTOR_ARRAY_SIZE), dtype='int')
    t1 = np.ndarray((K_MATRIX, VECTOR_ARRAY_SIZE), dtype='int')

    #apply power2round component-wise
    for i in range(K_MATRIX):
        for j in range(VECTOR_ARRAY_SIZE):
            t1[i][j], t0[i][j] = power_2_round(t[i][j])

    pk = pkEncode(rho, t1)
    tr = h_shake256(pk.tobytes(), 64)
    sk = skEncode(rho, k, tr, s1, s2, t0)
    return pk, sk

def ml_dsa_key_gen() -> Tuple[bitarray, bitarray]:
    '''
    Generates a public-private key pair. 
    '''
    #generate 256 bit seed
    seed = random.getrandbits(256) #change to approved RBG

    if seed == None: #needed?
        return None

    temp = seed.to_bytes(32, BYTEORDER)
    return ml_dsa_key_gen_internal(temp)

def ml_dsa_sign_internal(sk: bitarray, m: bitarray, rnd: bitarray) -> Any:
    rho, k, tr, s1, s2, t0 = skDecode(sk)

    s1_hat = [ NTT(s) for s in s1]
    s2_hat = [ NTT(s) for s in s2]
    t0_hat = [ NTT(s) for s in t0]

    a_hat = expand_a(rho)

    #prepend message w tr, convert to bytes, and hash
    message_bytes = (tr + m).tobytes()
    mu = h_shake256(message_bytes, 512)

    #compute private random seed
    private_seed_bytes = (k + rnd + mu).tobytes()
    rho_double_prime = h_shake256(private_seed_bytes, 512)

    kappa = 0
    z, h = None, None
    while z is None or h is None:
        print(f"Kappa: {kappa}")
        y = expand_mask(rho_double_prime, kappa)
        print(f"Kappa: {kappa}")

        #Hadamard product of A and NTT of y
        w_temp_prod = matrix_vector_ntt(a_hat, [NTT(x) for x in y])
        w = [NTT_inv(x) for x in w_temp_prod]

        #highbits() is applied componentwise to produce w1
        w1 = np.empty((K_MATRIX, VECTOR_ARRAY_SIZE), dtype='int')
        for i in range(K_MATRIX):
            for j in range(VECTOR_ARRAY_SIZE):
                w1[i][j] = high_bits(w[i][j])
  

        c_hash_seed_bytes = (mu + w1_encode(w1)).tobytes()
        c_hash = h_shake256(c_hash_seed_bytes, LAMBDA_COLLISION_STR // 4)
        c = sample_in_ball(c_hash)

        c_hat = NTT(c)

        cs1_prod = scalar_vector_ntt(c_hat, s1_hat)
        cs1 = [NTT_inv(sub) for sub in cs1_prod]

        cs2_prod = scalar_vector_ntt(c_hat, s2_hat)
        cs2 = [NTT_inv(sub) for sub in cs2_prod]

        z = add_vector_ntt(y, cs2)
        
        r0 = np.empty((K_MATRIX, VECTOR_ARRAY_SIZE), dtype='int')
        for i in range(K_MATRIX):
            for j in range(VECTOR_ARRAY_SIZE):
                r0[i][j] = low_bits(w[i][j] - cs2[i][j])
        print(f"z: {z.max()} >= {GAMMA_1_COEFFICIENT - BETA} or r0: {r0.max()} >= {GAMMA_2_LOW_ORDER_ROUND - BETA}")
        if z.max() >= (GAMMA_1_COEFFICIENT - BETA) or r0.max() >= (GAMMA_2_LOW_ORDER_ROUND - BETA):
            z, h = None, None
        else:
            ct0 = "ee"


def ml_dsa_sign(sk: bitarray, m: bitarray, ctx: bitarray) -> Any:
    #lmao figure out what return type is
    
    #context string cannot exceed 255 bytes
    if ctx.nbytes > 255:
        return None

    seed = random.getrandbits(256) #change to approved RBG

    temp = seed.to_bytes(32, BYTEORDER)

    rnd = new_bitarray()
    rnd.frombytes(temp)

    m_prime = integer_to_bits(0, 8) + integer_to_bits(int(len(ctx) / 8), 8) + ctx + m
    sigma = ml_dsa_sign_internal(sk, m_prime, rnd)

pk, sk = ml_dsa_key_gen()

ctx_b = bytes("testingshit", 'utf-8')
ctx = new_bitarray()
ctx.frombytes(ctx_b)

m = bitarray("10000111100101100")

sigma = ml_dsa_sign(sk, m, ctx)