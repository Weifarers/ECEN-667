import numpy as np
import matplotlib.pyplot as plt
import cmath


def newton_raphson(x, f, j):
    # Newton Raphson Method Function.
    # Inputs: x - Initial guess of the solution to the set of functions.
    #         f - The set of functions you're trying to solve for.
    #         j - The Jacobian of the set of functions.
    # Output: x - Updated solution to the set of functions.
    max_iter = 100
    epsilon = 0.001
    i = 0
    # We initialize the function evaluated at x independently so we can check when to exit the loop.
    fx = f(x)
    while (np.max(np.abs(f(x))) > epsilon) and i < max_iter:
        # Updates the value of x.
        x = x - np.dot(np.linalg.inv(j(x)), fx)
        # Evaluates f at the new x.
        fx = f(x)
        i += 1

    return x


def trapezoidal_f_of_x(xt, dt, f):
    # Defines the general form of the Implicit Trapezoidal Method that's fed into the Newton-Raphson Method.
    # Inputs: xt - The value of x(t)
    #         dt - Time step.
    #         f - The set of functions being evaluated.
    # Outputs: A matrix containing the trapezoidal function that still depends on x(t + dt)
    def trapezoidal_function(xtdt):
        # xtdt is the value of x(t + dt)
        return xtdt - xt - (dt/2) * (f(xt) + f(xtdt))
    return trapezoidal_function


def trapezoidal_jacobian(dt, j):
    # Defines the Jacobian of the Implicit Trapezoidal Method.
    # Inputs: j - The Jacobian of the original set of functions.
    #         dt - Time step.
    # Outputs: A matrix containing the jacobian that still depends on x(t + dt)
    def trapezoid_jac(xtdt):
        # xtdt is the value of x(t + dt)
        return  np.identity(xtdt.shape[0]) - (dt/2)*j(xtdt)
    return trapezoid_jac


def f_of_x(x):
    # Defines the set of equations we're trying to solve for.
    # This takes in a value x and returns a matrix containing the evaluated functions.
    x1 = x[0]
    x2 = x[1]
    f_matrix = np.array(
        [
            x2 * 2 * cmath.pi * 60,
            1.2/10
        ]
    )
    return f_matrix


def jacobian_of_f(x):
    # Defines the Jacobian of the set of equations.
    # This takes in a value x and returns a matrix containing the evaluated Jacobian.
    x1 = x[0]
    x2 = x[1]
    j_matrix = np.array(
        [
            [0, 2 *cmath.pi * 60],
            [0, 0]
        ]
    )
    return j_matrix


def main():
    # Initial and final time point.
    t0 = 0
    t_final = 0.04
    # Initializing the time step and initial guess.
    delta_t = 0.02
    x0 = np.array([0.107, 0])
    array_size = ((t_final - t0) / delta_t)
    # Initializing arrays to store guesses for plotting.
    iter_array = np.zeros(int(array_size))
    x1_array = np.zeros(int(array_size))
    x2_array = np.zeros(int(array_size))
    # We're initializing our guess to start at x0.
    x_temp = x0
    # We also keep track of iterations.
    x_count = 0
    for x_count in range(0, int(array_size)):
        # Storing values for reference.
        iter_array[x_count] = x_count
        x1_array[x_count] = x_temp[0]
        x2_array[x_count] = x_temp[1]
        temp_trapezoidal_f = trapezoidal_f_of_x(x_temp, delta_t, f=f_of_x)
        temp_jacobian_f = trapezoidal_jacobian(delta_t, j=jacobian_of_f)
        x_temp = newton_raphson(x_temp, temp_trapezoidal_f, temp_jacobian_f)

    print(x1_array, x2_array)

if __name__ == '__main__':
    main()
