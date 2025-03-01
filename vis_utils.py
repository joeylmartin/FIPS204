
from parametres import Q_MODULUS, K_MATRIX, VECTOR_ARRAY_SIZE, N_PRIVATE_KEY_RANGE
import numpy as np
import itertools
from typing import Tuple








def center_mod_q(vec):
    """
    Centers a vector in the range (-Q/2, Q/2) instead of (0, Q).
    """
    return np.where(vec > Q_MODULUS // 2, vec - Q_MODULUS, vec)

def sample_lattice_point(A: np.ndarray, S2: np.ndarray, coeff_range: int = N_PRIVATE_KEY_RANGE) -> Tuple[np.ndarray, np.ndarray]:
    # Sample integer coefficients from a bounded Gaussian or uniform distribution
 
    #a_reshape = A.reshape(K_MATRIX, L_MATRIX * VECTOR_ARRAY_SIZE) #flatten L matrix into 1D array, K x (256L) dimensions
    coefficient_values = list(range(-coeff_range, coeff_range + 1))
    coefficients = np.array(list(itertools.product(coefficient_values, repeat=K_MATRIX)))  # Num_combos Exhaustive combinations

    # Generate lattice points as integer combinations of basis vectors
    lattice_points = np.einsum('klm,nl->nkm', A, coefficients) % Q_MODULUS

    # Center modulo q
    lattice_points_mod = center_mod_q(lattice_points)

    # Expand S2 for all sampled points (Num_combos, K, 256)
    S2_expanded = np.repeat(S2[None, :, :], coefficients.shape[0], axis=0)

    # Apply error
    lattice_points_with_error = (lattice_points + S2_expanded) % Q_MODULUS
    lattice_points_with_error_mod = center_mod_q(lattice_points_with_error)
    
    # Flatten from (Num_combos, K, 256) to (Num_combos, K * 256)
    lattice_points_flat = lattice_points_mod.reshape(coefficients.shape[0], K_MATRIX * VECTOR_ARRAY_SIZE)
    lattice_points_with_error_flat = lattice_points_with_error_mod.reshape(coefficients.shape[0], K_MATRIX * VECTOR_ARRAY_SIZE)

    return lattice_points_flat, lattice_points_with_error_flat