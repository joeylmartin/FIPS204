import hashlib
from bitarray import bitarray
from math import ceil

from Crypto.Hash import SHAKE128, SHAKE256
from typing import Union

from .auxiliary_funcs import new_bitarray

HASH = Union[SHAKE128, SHAKE256]


def h_init() -> HASH:
    '''
    Return a SHAKE-256 hash object
    '''
    return SHAKE256.new()

def g_init() -> HASH:
    '''
    Return a SHAKE-128 hash object
    '''
    return SHAKE128.new()

def hash_absorb(ctx : HASH, data : bitarray) -> HASH:
    '''
    Injects data to be used in the absorbing phase of XOF and updates context ctx.
    '''
    ctx.update(bytes(data))
    return ctx

def hash_squeeze(ctx : HASH, bit_length : int) -> tuple[HASH, bitarray]:
    '''
    Extracts bit_length output bits produced during the squeezing phase of XOF
    and updates context ctx
    '''
    byte_length = bit_length // 8
    byte_output = ctx.read(byte_length)
    bits = new_bitarray()
    bits.frombytes(byte_output) #creates a bitarray from the byte_output

    return ctx, bits 

#Note: SHAKE-256 and SHAKE-128 are written using bit_length, but in 
#the original implementation, they use byte_length.

def h_shake256(seed: bytes, bit_length: int) -> bitarray:
    '''
    Extend bit string seed with SHAKE-256 XOF
    '''
    hash_obj = h_init()

    hash_obj = hash_absorb(hash_obj, seed)
    hash_obj, extended_seed_bitarray = hash_squeeze(hash_obj, bit_length)

    return extended_seed_bitarray

def h_shake128(seed: bytes, bit_length: int) -> bitarray:
    '''
    Extend bit string seed with SHAKE-128 XOF
    '''
    hash_obj = g_init()

    hash_obj = hash_absorb(hash_obj, seed)
    hash_obj, extended_seed_bitarray = hash_squeeze(hash_obj, bit_length)

    return extended_seed_bitarray