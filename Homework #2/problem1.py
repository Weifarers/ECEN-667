import numpy as np


def main():
    # Defining some constants for function building.
    f = 60
    rho = 120
    g = 0.1609347 * 10**-3

    # We build our equations for self inductance and mutual inductance.
    zii = zii_builder(f, rho, g)
    zij = zij_builder(f, rho, g)

    # Defining constants for use in calculation.
    gmr_falcon = 0.0523
    r_falcon = 0.0612
    gmr_partridge = 0.0217
    r_partridge = 0.350

    # Defining distances, also for calculation. Kersting has an interesting way of calculating distances using the
    # complex plane.
    # First, we define all the points in the complex plane, with the origin sitting below conductor A.
    # The order goes from left to right, starting with phase conductors, and then using ground conductors.
    d_list = np.array([40j, 12 + 40j, 24 + 40j, 5 + 55j, 19 + 55j])
    # Pre-allocating the matrix where we'll store our distances.
    d_matrix = np.zeros((5, 5))
    # Then, we can construct the distance matrix.
    for i in range(d_matrix.shape[0]):
        for j in range(d_matrix.shape[1]):
            # This is an if statement checker to catch diagonals, which we set to the GMR of the conductor.
            if i == j and i <= 2:
                d_matrix[i, j] = gmr_falcon
            elif i == j and i > 2:
                d_matrix[i, j] = gmr_partridge
            # Otherwise, we just take the difference between the two distances and take the magnitude.
            else:
                d_matrix[i, j] = abs(d_list[i] - d_list[j])

    # Pre-allocating the matrix where we'll store the impedances.
    z_matrix = 1j * np.zeros((5, 5))
    # Constructing the impedance matrix.
    for i in range(z_matrix.shape[0]):
        for j in range(z_matrix.shape[1]):
            # Checker to catch diagonals, so that we calculate self impedance terms.
            if i == j and i <= 2:
                z_matrix[i, j] = zii(d_matrix[i, j], r_falcon)
            elif i == j and i > 2:
                z_matrix[i, j] = zii(d_matrix[i, j], r_partridge)
            # Otherwise, we get the mutual inductance terms.
            else:
                z_matrix[i, j] = zij(d_matrix[i, j])

    # Now, we'll partition the matrix into its respective parts.
    z_a = z_matrix[0:3, 0:3]
    z_b = z_matrix[0:3, 3:5]
    z_c = z_matrix[3:5, 0:3]
    z_d = z_matrix[3:5, 3:5]

    # Now we construct the phase impedance matrix.
    z_p = z_a - np.dot(np.dot(z_b, np.linalg.inv(z_d)), z_c)

    # Then, we need as_matrix, a matrix that helps us convert from phase impedance to sequence impedance.
    # We define a_s; Python probably has a phasor domain input, but oh well. Rectangular works too.
    a_s = 1 * np.cos(120*(np.pi/180)) + 1j * np.sin(120*(np.pi/180))
    as_inv_matrix = (1/3) * np.array([[1, 1, 1],
                                  [1, a_s, a_s**2],
                                  [1, a_s**2, a_s]])

    z_s = np.dot(np.dot(as_inv_matrix, z_p), np.linalg.inv(as_inv_matrix))
    print(z_p)
    print(z_s)


def zii_builder(f, rho, g):
    # Returns the modified Carson's Equation for self-impedance. We make this flexible for testing purposes.
    # Inputs: f - the frequency
    #         rho - resistivity in ohms per mile
    #         g = 0.1609347*10^-3, some constant derived.
    # Outputs: A function zii_eq that depends on gmr and ri.
    def zii_eq(gmr, ri):
        return ri + (np.pi**2 * g * f) + (4 * np.pi * g * f) * (np.log(1/gmr) + 7.6786 + (1/2) * np.log(rho/f)) * 1j
    return zii_eq


def zij_builder(f, rho, g):
    # Returns the modified Carson's Equation for mutual impedance. We make this flexible for testing purposes.
    # Inputs: f - the frequency
    #         rho - resistivity in ohms per mile
    #         g = 0.1609347*10^-3, some constant derived.
    # Outputs: A function zij_eq that depends on dij.
    def zij_eq(dij):
        return (np.pi**2 * g * f) + (4 * np.pi * g * f) * (np.log(1/dij) + 7.6786 + (1/2) * np.log(rho/f)) * 1j
    return zij_eq


if __name__ == '__main__':
    main()
