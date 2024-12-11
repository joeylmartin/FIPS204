import math

BYTEORDER='little' #CHECK IF ENDIANNESS MESSES STUFF UP!!


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

#each vector consists of 256 components
#and so do polynomials. maybe change?
VECTOR_ARRAY_SIZE = 256 