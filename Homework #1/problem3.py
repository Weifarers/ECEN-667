import numpy as np
import matplotlib.pyplot as plt


def main():
    # Initializations.
    l_prime = 2.18 * 10**-3
    c_prime = 0.0136 * 10**-6
    d = 225
    r_l = 300
    r_s = 5

    # We initialize the voltage source as a function of time. Thanks to Brandon for reminding me this kind of thing
    # works in Python!
    def v_s(time):
        return 188000*np.cos(2 * np.pi * 60 * time)

    # Calculating the first set of constants.
    z_c = np.sqrt(l_prime / c_prime)
    v_p = 1 / np.sqrt(l_prime * c_prime)
    print(z_c, v_p)

    # Getting some time variables.
    tau = d / v_p
    print(tau)
    n = 6
    delta_t = (1 / n) * tau
    t_end = 0.04
    # We're constructing a list of points to evaluate through here.
    t_points = np.arange(0, t_end, delta_t)

    # Initializing a bunch of arrays to store values. The sizes will depend on the number of
    # time points we're considering.
    i_k = np.zeros(t_points.shape[0])
    i_m = np.zeros(t_points.shape[0])
    i_s = np.zeros(t_points.shape[0])
    i_l = np.zeros(t_points.shape[0])
    v_k = np.zeros(t_points.shape[0])
    v_m = np.zeros(t_points.shape[0])

    # Also initializing a counter.
    i = 0

    # Now we'll iterate through all the time points.
    for t in t_points:
        # First we need to evaluate i_k and i_m. These depend on points in the past.
        # But, we need a check for instances where the time is not yet at tau.
        if t >= tau:
            i_k[i] = i_l[i - n] - (v_m[i - n] / z_c)
            i_m[i] = i_s[i - n] + (v_k[i - n] / z_c)
        # Now we can get the remaining values.
        # We set an if statement to catch the first time instance, so we don't overwrite the values calculated
        # above.
        i_s[i] = (v_s(t) + z_c*i_k[i])/(r_s + z_c)
        v_k[i] = v_s(t) - r_s*i_s[i]
        i_l[i] = i_m[i] * (z_c/(z_c + r_l))
        v_m[i] = z_c*(i_m[i] - i_l[i])

        # Updating our counter.
        i += 1

    # Now we make two different plots; one for current, and one for voltage.
    plt.figure()
    plt.plot(t_points, v_k, 'r', t_points, v_m, 'b')
    plt.title('Sending and Receiving End Voltages')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend(['Sending', 'Receiving'], loc='upper right')
    plt.savefig('voltage.png')

    plt.figure()
    plt.plot(t_points, i_s, 'r', t_points, i_l, 'b')
    plt.title('Sending and Receiving End Currents')
    plt.xlabel('Time (s)')
    plt.ylabel('Current (A)')
    plt.legend(['Sending', 'Receiving'], loc='upper right')
    plt.savefig('current.png')

    plt.show()


if __name__ == '__main__':
    main()

