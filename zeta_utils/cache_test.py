import pickle



q = 8380417
zeta = 1753

# Function to compute bit-reversal of k
def bit_reverse(k, bit_width=8):

    binary_k = format(k, '0{}b'.format(bit_width))

    reversed_binary_k = binary_k[::-1]

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

precomputed_values.append(1)  # zeta^0 = 1
for k in range(1, 256):
    zeta_brv_k = compute_zeta_brv_k(k)
    precomputed_values.append(zeta_brv_k)

# Optional: Save the values to a text file
with open('zeta_brv_k_cache.pkl', 'wb') as file:
    pickle.dump(precomputed_values, file)


print("Precomputed values of zeta^{brv(k)} have been stored.")
