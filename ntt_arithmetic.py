import numpy as np
from parametres import *

def add_ntt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    '''
    Add two NTT polynomials
    '''
    c = np.empty(VECTOR_ARRAY_SIZE)
    for i in range(VECTOR_ARRAY_SIZE):
        c[i] = (a[i] + b[i]) % Q_MODULUS
    return c 

def multiply_ntt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    '''
    Multiply two NTT polynomials
    '''
    c = np.empty(VECTOR_ARRAY_SIZE)
    for i in range(VECTOR_ARRAY_SIZE):
        c[i] = (a[i] * b[i]) % Q_MODULUS
    return c

def add_vector_ntt(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    '''
    Add two NTT vectors
    '''
    u = np.empty((L_MATRIX, VECTOR_ARRAY_SIZE))
    for i in range(L_MATRIX):
        u[i] = add_ntt(v[i], w[i])
    return u

def scalar_vector_ntt(c: int, v: np.ndarray) -> np.ndarray:
    '''
    Computes the product c ∘ v of a scalar c and a vector v over Tq.
    '''
    w = np.empty((L_MATRIX, VECTOR_ARRAY_SIZE))
    for i in range(L_MATRIX):
        w[i] = multiply_ntt(c, v[i])
    return w

def matrix_vector_ntt(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    '''
    Computes the product M ∘ v of a matrix M and a vector v over Tq.
    '''
    w = np.zeros((K_MATRIX, VECTOR_ARRAY_SIZE))
    for i in range(K_MATRIX):
        for j in range(L_MATRIX):
            w[i] = add_ntt(w[i], multiply_ntt(M[i][j], v[j]))
    return w