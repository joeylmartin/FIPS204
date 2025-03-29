
from fips_204.external_funcs import ml_dsa_key_gen, ml_dsa_sign, ml_dsa_verify
from fips_204.internal_funcs import NTT, NTT_inv, pkDecode, sample_in_ball, sigDecode, skDecode, expand_a
import fips_204.internal_funcs as internal_funcs
from fips_204.auxiliary_funcs import new_bitarray, use_hint
from fips_204.ntt_arithmetic import add_vector_ntt, matrix_vector_ntt, scalar_vector_ntt
from fips_204.parametres import BYTEORDER, D_DROPPED_BITS, K_MATRIX, VECTOR_ARRAY_SIZE
import os
import numpy as np
import random


pk, sk = ml_dsa_key_gen()

ctx_b = os.urandom(255)
ctx = new_bitarray()
ctx.frombytes(ctx_b)

seed = random.getrandbits(256) #change to approved RBG
s_b = seed.to_bytes(32, BYTEORDER)
mb = new_bitarray()
mb.frombytes(s_b)

rho, t1 = pkDecode(pk)

def update_on_message(message):
    global a, s1, s2, y, c, z, w1_prime, w, w_a, a_rq, t
    sig = ml_dsa_sign(sk, message, ctx) 
    ver = ml_dsa_verify(pk, mb, sig, ctx)

    rho, k, tr, s1, s2, t0 = skDecode(sk)
    a = expand_a(rho)

    #non ntt variant
    a_rq = [[NTT_inv(poly) for poly in vec] for vec in a]

    c_hash, z, h = sigDecode(sig)
    c = sample_in_ball(c_hash)
    c_hat = NTT(c)

    s1_hat = [ NTT(s) for s in s1]
    cs1_prod = scalar_vector_ntt(c_hat, s1_hat) 
    cs1 = np.array([NTT_inv(sub) for sub in cs1_prod])
        
    y = z - cs1
    w_temp_prod = matrix_vector_ntt(a, [NTT(x) for x in y])
    w = [NTT_inv(x) for x in w_temp_prod]

    ntt_product = matrix_vector_ntt(a, [NTT(x) for x in s1])
    t = [NTT_inv(x) for x in ntt_product] + s2
    
    s2_hat = [ NTT(s) for s in s2]
    cs2_prod = scalar_vector_ntt(c_hat, s2_hat)
    cs2 = np.array([NTT_inv(sub) for sub in cs2_prod])

    #Get W1' (without hint or T rounding just by subbing cs2)
    w_a = w - cs2

update_on_message(mb)



