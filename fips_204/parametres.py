import math
from enum import Enum

BYTEORDER='little' #CHECK IF ENDIANNESS MESSES STUFF UP!!

#PARAMETRES e.g
class Strength(Enum):
    ML_DSA_44 = 1
    ML_DSA_65 = 2
    MLA_DSA_87 = 3

STRENGTH = Strength.ML_DSA_44

match STRENGTH:
    case Strength.ML_DSA_44:
        Q_MODULUS = 8380417 
        D_DROPPED_BITS = 13 
        TAU_ONES = 39
        LAMBDA_COLLISION_STR = 128
        GAMMA_1_COEFFICIENT = int(math.pow(2, 17))
        GAMMA_2_LOW_ORDER_ROUND = (Q_MODULUS - 1)//88
        K_MATRIX = 4
        L_MATRIX = 4
        N_PRIVATE_KEY_RANGE = 2
        BETA = TAU_ONES * N_PRIVATE_KEY_RANGE
        W_MAX_HINT_ONES = 80
    case Strength.ML_DSA_65:
        Q_MODULUS = 8380417 
        D_DROPPED_BITS = 13 
        TAU_ONES = 49
        LAMBDA_COLLISION_STR = 192
        GAMMA_1_COEFFICIENT = int(math.pow(2, 19))
        GAMMA_2_LOW_ORDER_ROUND = (Q_MODULUS - 1)//32
        K_MATRIX = 6
        L_MATRIX = 5
        N_PRIVATE_KEY_RANGE = 4
        BETA = TAU_ONES * N_PRIVATE_KEY_RANGE
        W_MAX_HINT_ONES = 55
    case Strength.ML_DSA_87:
        Q_MODULUS = 8380417 
        D_DROPPED_BITS = 13 
        TAU_ONES = 60
        LAMBDA_COLLISION_STR = 256
        GAMMA_1_COEFFICIENT = int(math.pow(2, 19))
        GAMMA_2_LOW_ORDER_ROUND = (Q_MODULUS - 1)//32
        K_MATRIX = 8
        L_MATRIX = 7
        N_PRIVATE_KEY_RANGE = 2
        BETA = TAU_ONES * N_PRIVATE_KEY_RANGE
        W_MAX_HINT_ONES = 75
    #case _:

#each vector consists of 256 components
#and so do polynomials. maybe change?
VECTOR_ARRAY_SIZE = 256 