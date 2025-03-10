from bitarray import bitarray
from typing import Tuple
import random
from .internal_funcs import *
from .parametres import BYTEORDER

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

def ml_dsa_sign(sk: bitarray, m: bitarray, ctx: bitarray) -> bitarray:
    '''
        rnd_override (32 byte long bitarray) argument not included in FIPS-204 scheme. 
        Used to fix value for testing.
    '''
    #context string cannot exceed 255 bytes
    if ctx.nbytes > (255):
        raise Exception("Context string can only be 255 bytes long!")
    
    seed = random.getrandbits(256) #change to approved RBG

    s_b = seed.to_bytes(32, BYTEORDER)
    rnd = new_bitarray()
    rnd.frombytes(s_b)

    m_prime = integer_to_bits(0, 8) + integer_to_bits(int(len(ctx) / 8), 8) + ctx + m
    sigma = ml_dsa_sign_internal(sk, m_prime, rnd)
    return sigma

def ml_dsa_verify(pk: bitarray, m: bitarray, sigma: bitarray, ctx: bitarray) -> bool:
    '''
    Verifies signature sigma for a message M.
    '''
    if ctx.nbytes > (255):
        raise Exception("Context string can only be 255 bytes long!")

    m_prime = integer_to_bits(0, 8) + integer_to_bits(ctx.nbytes, 8) + ctx + m
    return ml_dsa_verify_internal(pk, m_prime, sigma)