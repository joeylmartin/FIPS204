import hashlib
from bitarray import bitarray
from hashlib import _hashlib
from _hashlib import HASH

def h() -> HASH:
    '''
    Return a SHAKE-256 hash object
    '''
    return hashlib.shake_256()

def g() -> HASH:
    '''
    Return a SHAKE-128 hash object
    '''
    return hashlib.shake_128()

def hash_absorb(ctx : HASH, data : bitarray) -> HASH:
    '''
    Injects data to be used in the absorbing phase of XOF and updates context ctx.
    '''
    ctx.update(data)
    return ctx

def hash_squeeze(ctx : HASH, bit_length : int) -> tuple[HASH, bitarray]:
    '''
    Extracts bit_length output bits produced during the squeezing phase of XOF
    and updates context ctx
    '''
    byte_length = bit_length // 8
    byte_output = ctx.digest(bit_length)
    bits = bitarray()
    bits.frombytes(byte_output)
    return ctx, bits

def h_shake256(seed: bytes, bit_length: int) -> bitarray:
    '''
    Extend bit string seed with SHAKE-256 XOF
    '''
    hash_obj = h()

    hash_obj = hash_absorb(hash_obj, seed)
    hash_obj, extended_seed_bitarray = hash_squeeze(hash_obj, bit_length)

    return extended_seed_bitarray

def h_shake128(seed: bytes, bit_length: int) -> bitarray:
    '''
    Extend bit string seed with SHAKE-128 XOF
    '''
    hash_obj = g()

    hash_obj = hash_absorb(hash_obj, seed)
    hash_obj, extended_seed_bitarray = hash_squeeze(hash_obj, bit_length)

    return extended_seed_bitarray