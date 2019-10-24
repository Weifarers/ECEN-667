import numpy as np
import matplotlib.pyplot as plt


def main():
    # Initial and final time point.
    t0 = 0
    t_final = 10
    # Initializing the time step and initial guess.
    delta_t = 0.01
    x0 = np.array([5, 5, 5])
    array_size = np.ceil((t_final - t0)/delta_t)
    print(array_size)
    # Initializing arrays to store guesses for plotting.
    iter_array = np.zeros(int(array_size))
    x1_array = np.zeros(int(array_size))
    x2_array = np.zeros(int(array_size))
    x3_array = np.zeros(int(array_size))
    # We're initializing our guess to start at x0.
    x_temp = x0

    for x_count in range(0, int(array_size)):
        # Storing values for reference.
        iter_array[x_count] = x_count
        x1_array[x_count] = x_temp[0]
        x2_array[x_count] = x_temp[1]
        x3_array[x_count] = x_temp[2]
        # Calculating k1/k2.
        k1 = delta_t * f_of_x(x_temp)
        k2 = delta_t * (f_of_x(x_temp + k1))
        # Updating the value of the guess.
        x_temp = x_temp + (k1 + k2)/2

    # Plotting the solutions.
    plt.plot(iter_array, x1_array, 'r', iter_array, x2_array, 'b', iter_array, x3_array, 'g')
    # Setting the options for the plots.
    plt.title('Plot of Solution for RK2 Method')
    plt.xlabel('Iteration #')
    plt.ylabel('Value')
    plt.legend(('x1', 'x2', 'x3'), loc='upper right')
    plt.savefig('RK2.png')
    plt.show()


def f_of_x(x):
    # Evaluating the function.
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    f = np.array([10 * (x2 - x1), x1 * (28 - x3) - x2, x1*x2 - (8/3)*x3])
    return f


if __name__ == '__main__':
    main()
