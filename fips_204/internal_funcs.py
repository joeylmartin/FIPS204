import math
import pickle
from typing import Tuple
import numpy as np
from bitarray import bitarray, _set_default_endian
from .parametres import *
from .ntt_arithmetic import *
from .auxiliary_funcs import *
from .hash_funcs import h_shake256, h_init, hash_absorb, hash_squeeze, g_init

_set_default_endian(BYTEORDER)

def load_zeta_brv_cache(cache_file='zeta_utils/zeta_brv_k_cache.pkl'):
    '''
    Load precomputed values of zeta from a cache file.
    '''
    with open(cache_file, 'rb') as file:
        zeta_brv = pickle.load(file)
    return zeta_brv

cached_zeta_brv = load_zeta_brv_cache()


def NTT(w: np.ndarray) -> np.ndarray:
    '''
    Converts an element of Rq into Tq with the Number
    Theoretic Transform (NTT).
    '''
    w_hat = w.copy() #d swap
    
    m = 0
    len = 128
    while len >= 1:
        start = 0
        while start < 256:
            m += 1
            z = cached_zeta_brv[m]
            for j in range(start, start + len):
                t = (z * w_hat[j + len]) % Q_MODULUS
                w_hat[j + len] = (w_hat[j] - t) % Q_MODULUS
                w_hat[j] = (w_hat[j] + t) % Q_MODULUS
            start += (2 * len)
        len //= 2
    return w_hat

def NTT_inv(w_hat: np.ndarray) -> np.ndarray:
    '''
    Converts an element of Tq into Rq.
    '''
    w = w_hat.copy() #d swap
    
    m = 256
    len = 1
    while len < 256:
        start = 0
        while start < 256:
            m -= 1
            z = -cached_zeta_brv[m]
            for j in range(start, start + len):
                t = w[j]
                w[j] = (t + w[j + len]) % Q_MODULUS
                w[j + len] = (t - w[j + len]) % Q_MODULUS
                w[j + len] = (z * w[j + len]) % Q_MODULUS
            start += 2 * len
        len *= 2

    f = 8347681
    w = np.array([(f * j) % Q_MODULUS for j in w]) 

    return w

def pkEncode(rho : bitarray, t1 : np.ndarray) -> bitarray:
    '''
    Encodes a public key for ML-DSA into a bit string.
    Where T1 is K polynomials (k x 256 np array)
    '''
    pk = rho.copy()
    bit_len = (1 << ((Q_MODULUS - 1).bit_length() - D_DROPPED_BITS)) - 1

    for i in range(K_MATRIX):
        pk += simple_bit_pack(t1[i], bit_len)
    return pk

def pkDecode(pk: bitarray) -> Tuple[bitarray, np.ndarray]:
    '''
    Decodes a public key for ML-DSA from a bit string.
    '''
    rho = pk[:256]

    z_len = (Q_MODULUS - 1).bit_length() - D_DROPPED_BITS #

    t1 = np.ndarray((K_MATRIX, VECTOR_ARRAY_SIZE), dtype='int64')
    index = 256
    for i in range(K_MATRIX):
        data = pk[index : index + (32 * 8 * z_len)]
        t1[i] = simple_bit_unpack(data, int(math.pow(2,z_len) - 1))
        index += 32 * 8 * z_len
    
    return rho, t1

def skEncode(rho : bitarray, k : bitarray, tr : bitarray, s1 : np.ndarray, s2 : np.ndarray, t0 : np.ndarray) -> bitarray:
    '''
    Encodes the components of a secret key into one bitarray.
    '''
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

    y_z_len = 8 * 32 * ((2 * N_PRIVATE_KEY_RANGE).bit_length())


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

    s1 = np.ndarray((L_MATRIX,VECTOR_ARRAY_SIZE), dtype='int64')
    s2 = np.ndarray((K_MATRIX,VECTOR_ARRAY_SIZE), dtype='int64')
    t0 = np.ndarray((K_MATRIX,VECTOR_ARRAY_SIZE), dtype='int64')

    for i in range(L_MATRIX):
        s1[i] = bit_unpack(y[i], N_PRIVATE_KEY_RANGE, N_PRIVATE_KEY_RANGE)
    for i in range(K_MATRIX):
        s2[i] = bit_unpack(z[i], N_PRIVATE_KEY_RANGE, N_PRIVATE_KEY_RANGE)
    
    lower_bound = int(math.pow(2, D_DROPPED_BITS - 1) - 1)
    for i in range(K_MATRIX):
        t0[i] = bit_unpack(w[i], lower_bound, lower_bound + 1)
    
    return rho, k, tr, s1, s2, t0



def sigEncode(c_hash : bitarray, z : np.ndarray, h : np.ndarray) -> bitarray:
    '''
    Encodes a signature for ML-DSA into a bit string.
    '''
    sig = c_hash.copy()
    for i in range(L_MATRIX):
        sig += bit_pack(z[i], GAMMA_1_COEFFICIENT - 1, GAMMA_1_COEFFICIENT)
    sig += hint_bit_pack(h)
    return sig

def sigDecode(sig : bitarray) -> Tuple[bitarray, np.ndarray, np.ndarray]:
    '''
    Decodes a signature for ML-DSA from a bit string.
    '''
    c_hash = sig[:2 * LAMBDA_COLLISION_STR]
    z = np.ndarray((L_MATRIX, VECTOR_ARRAY_SIZE), dtype='int64')
    h = np.ndarray((K_MATRIX, VECTOR_ARRAY_SIZE), dtype='int64')

    #Z is a L x z_len array
    z_len = 8 * 32 * (1 + (GAMMA_1_COEFFICIENT - 1).bit_length())

    index = 2 * LAMBDA_COLLISION_STR
    for i in range(L_MATRIX):
        data = sig[index : index + z_len]
        z[i] = bit_unpack(data, GAMMA_1_COEFFICIENT - 1, GAMMA_1_COEFFICIENT)
        index += z_len

    y = sig[index : index + ((W_MAX_HINT_ONES + K_MATRIX) * 8)]
    h = hint_bit_unpack(y)
    return c_hash, z, h

def w1_encode(w1: np.ndarray) -> bitarray:
    '''
    Encodes a polynomial w1 into a bit string.
    '''
    w_hat = new_bitarray()
    bit_pack_bound= int(((Q_MODULUS - 1)/(2 * GAMMA_2_LOW_ORDER_ROUND)) - 1)
    for i in range(K_MATRIX):
        w_hat += simple_bit_pack(w1[i], bit_pack_bound)
    return w_hat



#PSEUDO-RANDOM SAMPLING FUNCTIONS

def sample_in_ball(rho: bitarray) -> np.ndarray:

    c = np.zeros(VECTOR_ARRAY_SIZE, dtype='int64')

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
        temp2 = int(math.pow(-1,temp))
        c[j] = temp2

    return c

def rej_ntt_poly(rho: bitarray) -> np.ndarray:
    '''
    Samples an polynomial in Tq,
    where Rho is a 272 bit string. 
    '''

    j = 0
    a = np.empty(VECTOR_ARRAY_SIZE, dtype='int64')

    ctx = g_init()
    ctx = hash_absorb(ctx, rho)

    while j < 256:
        ctx, s_bit = hash_squeeze(ctx, 24)

        byte_0 = bitarr_get_byte(s_bit, 0)        
        byte_1 = bitarr_get_byte(s_bit, 1)             
        byte_2 = bitarr_get_byte(s_bit, 2)     

        res = coeff_from_three_bytes(byte_0,byte_1, byte_2)
        if res is not None:
            a[j] = res
            j += 1

    return a

def rej_bounded_poly(rho: bitarray) -> np.ndarray:
    '''
    Samples an element a ∈ Rq with coeffcients in [−η,η] computed via rejection sampling from rho.
    Rho is a 528 bit string. 
    '''
    j = 0
    a = np.empty(VECTOR_ARRAY_SIZE)

    ctx = h_init()
    ctx = hash_absorb(ctx, rho)

    while j < VECTOR_ARRAY_SIZE:
        ctx, z_bit = hash_squeeze(ctx, 8)
        z = ba2int(z_bit) #conv bitarray to int
        z0 = coeff_from_half_byte(z % 16)
        z1 = coeff_from_half_byte(z // 16)

        if z0 is not None:
            a[j] = z0
            j += 1
        if z1 is not None and j < VECTOR_ARRAY_SIZE:
            a[j] = z1
            j += 1

    return a


def expand_a(rho: bitarray) -> np.ndarray:
    '''
    Samples a k x l matrix A of elements of Tq; where Rho is a 256 bit string
    '''
    matrix = np.zeros((K_MATRIX, L_MATRIX, VECTOR_ARRAY_SIZE), dtype='int64')

    for r in range(0, K_MATRIX):
        for s in range(0, L_MATRIX): 
            rho_prime = rho + integer_to_bits(s, 8) + integer_to_bits(r, 8)
            matrix[r][s] = rej_ntt_poly(rho_prime)
    
    return matrix


def expand_s(rho: bitarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Samples vectors s1 ∈ Rlq and s2 ∈ Rkq, each with 
    coeffcients in the interval [−η,η].
    '''
    s1, s2 = np.empty((L_MATRIX, VECTOR_ARRAY_SIZE), dtype='int64'), np.empty((K_MATRIX, VECTOR_ARRAY_SIZE), dtype='int64')
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
    y = np.empty((L_MATRIX,VECTOR_ARRAY_SIZE), dtype='int64')

    c = (GAMMA_1_COEFFICIENT - 1).bit_length() + 1

    for r in range(L_MATRIX):
        rho_prime = rho + integer_to_bits(mu + r, 16)
        v = h_shake256(rho_prime.tobytes(), 32 * 8 * c)
        y[r] = bit_unpack(v, GAMMA_1_COEFFICIENT-1, GAMMA_1_COEFFICIENT)
    
    return y

def ml_dsa_key_gen_internal(seed: bytes) -> Tuple[bitarray, bitarray]:
    '''
    Generates a public-private key pair.
    '''
    extended_seed = h_shake256(seed + integer_to_bytes(K_MATRIX, 1) + integer_to_bytes(L_MATRIX, 1), 1024)

    rho = extended_seed[:256] #first 256 bits
    rho_prime = extended_seed[256:768] #middle 512 bits
    k = extended_seed[768:] #last 256 bits
    
    
    #Create Matrix A, and vectors s1 (secret) and s2 (error)
    a = expand_a(rho)
    s1, s2 = expand_s(rho_prime)

    #product of A and NTT of S1
    ntt_product = matrix_vector_ntt(a, [NTT(x) for x in s1])
    t = ([NTT_inv(x) for x in ntt_product] + s2) % Q_MODULUS


    t0 = np.ndarray((K_MATRIX, VECTOR_ARRAY_SIZE), dtype='int64')
    t1 = np.ndarray((K_MATRIX, VECTOR_ARRAY_SIZE), dtype='int64')

    #apply power2round component-wise
    for i in range(K_MATRIX):
        for j in range(VECTOR_ARRAY_SIZE):
            t1[i][j], t0[i][j] = power_2_round(t[i][j])

    pk = pkEncode(rho, t1)
    tr = h_shake256(pk.tobytes(), 64 * 8)
    sk = skEncode(rho, k, tr, s1, s2, t0)
    return pk, sk

def ml_dsa_sign_internal(sk: bitarray, m: bitarray, rnd: bitarray) -> bitarray:
    '''
    Internal function to sign a message, given sk, message, and a random seed rnd.
    '''
    rho, k, tr, s1, s2, t0 = skDecode(sk)

    s1_hat = [ NTT(s) for s in s1]
    s2_hat = [ NTT(s) for s in s2]
    t0_hat = [ NTT(s) for s in t0]

    a_hat = expand_a(rho)

    #prepend message w tr, convert to bytes, and hash
    message_bytes = (tr + m).tobytes()
    mu = h_shake256(message_bytes, 64 * 8)

    #compute private random seed
    private_seed_bytes = (k + rnd + mu).tobytes()
    rho_double_prime = h_shake256(private_seed_bytes, 64 * 8)

    kappa = 0
    z, h = None, None
    
    while z is None or h is None:
        y = expand_mask(rho_double_prime, kappa)

        #Hadamard product of A and NTT of y
        w_temp_prod = matrix_vector_ntt(a_hat, [NTT(x) for x in y])
        w = [NTT_inv(x) for x in w_temp_prod]

        #highbits() is applied componentwise to produce w1
        w1 = np.empty((K_MATRIX, VECTOR_ARRAY_SIZE), dtype='int64')
        for i in range(K_MATRIX):
            for j in range(VECTOR_ARRAY_SIZE):
                w1[i][j] = high_bits(w[i][j])
  

        c_hash_seed_bytes = (mu + w1_encode(w1)).tobytes()
        c_hash = h_shake256(c_hash_seed_bytes, 2 * LAMBDA_COLLISION_STR)
        c = sample_in_ball(c_hash)
        c_hat = NTT(c)

        cs1_prod = scalar_vector_ntt(c_hat, s1_hat) 
        cs1 = np.array([NTT_inv(sub) for sub in cs1_prod])
    
        cs2_prod = (c_hat * s2_hat) % Q_MODULUS
        cs2 = np.array([NTT_inv(sub) for sub in cs2_prod])

        z = (y + cs1) % Q_MODULUS
        
        r0 = np.empty((K_MATRIX, VECTOR_ARRAY_SIZE), dtype='int64')
        for i in range(K_MATRIX):
            for j in range(VECTOR_ARRAY_SIZE):
                r0[i][j] = low_bits(w[i][j] - cs2[i][j])
        
        z_inf = get_vector_infinity_norm(z)
        r0_inf = get_vector_infinity_norm(r0)
        if (z_inf >= (GAMMA_1_COEFFICIENT - BETA)) or (r0_inf >= (GAMMA_2_LOW_ORDER_ROUND - BETA)): 
            z, h = None, None
        else:
            ct0_prod = (c_hat * t0_hat) % Q_MODULUS
            ct0 = np.array([NTT_inv(sub) for sub in ct0_prod])
            
            h = np.zeros((K_MATRIX, VECTOR_ARRAY_SIZE), dtype='int64')
            for i in range(K_MATRIX):
                for j in range(VECTOR_ARRAY_SIZE):
                    h[i][j] = make_hint(-ct0[i][j], w[i][j] - cs2[i][j] + ct0[i][j])

            num_ones = np.count_nonzero(h == 1)
            ct0_inf = get_vector_infinity_norm(ct0)
            if ct0_inf >= GAMMA_2_LOW_ORDER_ROUND or num_ones > W_MAX_HINT_ONES:
                z, h = None, None

        kappa += L_MATRIX

    z_mod = mod_pm_vector(z, Q_MODULUS)
    sig = sigEncode(c_hash, z_mod, h)
    return sig

def ml_dsa_verify_internal(pk: bitarray, m: bitarray, sigma: bitarray) -> bool: 
    '''
    Internal function to verify a signature sigma for a message M.
    '''
    rho, t1 = pkDecode(pk)
    c_hash, z, h = sigDecode(sigma)
    if h is None:
        return False

    a = expand_a(rho)
    tr = h_shake256(pk.tobytes(), 64 * 8)
    mu = h_shake256((tr + m).tobytes(), 64 * 8)
    c = sample_in_ball(c_hash)
    
    d_p2 = 1 << D_DROPPED_BITS

    w_a1 = matrix_vector_ntt(a, [NTT(x) for x in z])
    w_a2 = (NTT(c) * [NTT(d_p2 * x) for x in t1]) % Q_MODULUS
    w_temp_prod = (w_a1 - w_a2) % Q_MODULUS

    w_a = np.array([NTT_inv(x) for x in w_temp_prod])

    w1_prime = np.zeros((K_MATRIX, VECTOR_ARRAY_SIZE), dtype='int64')
    for i in range(K_MATRIX):
        for j in range(VECTOR_ARRAY_SIZE):
            w1_prime[i][j] = use_hint(h[i][j], w_a[i][j])

    c_hash_prime = h_shake256((mu + w1_encode(w1_prime)).tobytes(), 2 * LAMBDA_COLLISION_STR)

    return (get_vector_infinity_norm(z) < (GAMMA_1_COEFFICIENT - BETA)) and (c_hash == c_hash_prime)