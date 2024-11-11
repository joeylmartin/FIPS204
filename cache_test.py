import pickle


def load_zeta_brv_cache(cache_file='zeta_brv_k_cache.pkl'):
    with open(cache_file, 'rb') as file:
        zeta_brv = pickle.load(file)
    return zeta_brv

# Example usage
cached_zeta_brv = load_zeta_brv_cache()

'''
# Define modulus q and primitive root zeta
q = 8380417
zeta = 1753

# Function to compute bit-reversal of k
def bit_reverse(k, bit_width=8):
    # Format k as binary with leading zeros
    binary_k = format(k, '0{}b'.format(bit_width))
    # Reverse the bits
    reversed_binary_k = binary_k[::-1]
    # Convert back to integer
    brv_k = int(reversed_binary_k, 2)
    return brv_k

# Function to compute zeta^(brv(k)) mod q
def compute_zeta_brv_k(k):
    brv_k = bit_reverse(k)
    # Compute zeta^(brv(k)) mod q
    zeta_brv_k = pow(zeta, brv_k, q)
    return zeta_brv_k

# Precompute and store the values
precomputed_values = []

precomputed_values.append(0)  # zeta^0 = 1
for k in range(1, 256):
    zeta_brv_k = compute_zeta_brv_k(k)
    precomputed_values.append(zeta_brv_k)

# Optional: Save the values to a text file
with open('zeta_brv_k_cache.pkl', 'wb') as file:
    pickle.dump(precomputed_values, file)
'''

print("Precomputed values of zeta^{brv(k)} have been stored.")
