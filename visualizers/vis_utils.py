
from fips_204.parametres import Q_MODULUS, K_MATRIX, VECTOR_ARRAY_SIZE, N_PRIVATE_KEY_RANGE
import numpy as np
import itertools
from typing import Tuple
from abc import ABC, abstractmethod


class DemoPage(ABC):
    @abstractmethod
    def get_html(self):
        '''
        Returns a Dash HTML object that represents the subpage.
        '''
        pass
    @abstractmethod
    def register_callbacks(self, app):
        '''
        Allows us to pass the global Dash app object to the demo page,
        so that we can register the object's local callbacks to the global app.
        '''
        pass



def center_mod_q(vec):
    """
    Centers a vector in the range (-Q/2, Q/2) instead of (0, Q).
    """
    return np.where(vec > Q_MODULUS // 2, vec - Q_MODULUS, vec)

def sample_lattice_point(A: np.ndarray, S2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given Matrix A (representing basis vectors for a lattice), and error S2(K_MATRIX x 256), generate points on Lattice A,
    created by all basis combinations with coefficients in the range (-N_PRIVATE_KEY_RANGE, N_PRIVATE_KEY_RANGE).
    Returns lattice points, and points with S2 applied. Both are centered modulo Q. Returns flattened versions of these points(K_MATRIX x 256).
    Number of lattice points generated is (2 * N_PRIVATE_KEY_RANGE + 1) ^ K_MATRIX.
    """
    coefficient_values = list(range(-N_PRIVATE_KEY_RANGE, N_PRIVATE_KEY_RANGE + 1))
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