import numpy as np
import matplotlib.pyplot as plt


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
        return np.identity(xtdt.shape[0]) - (dt/2)*j(xtdt)
    return trapezoid_jac


def f_of_x(x):
    # Defines the set of equations we're trying to solve for.
    # This takes in a value x and returns a matrix containing the evaluated functions.
    x1 = x[0]
    x2 = x[1]
    f_matrix = np.array(
        [
            (2/3)*x1 - (4/3)*x1*x2,
            x1*x2 - x2
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
            [(2/3) - (4/3)*x2, -(4/3)*x1],
            [x2, x1 - 1]
        ]
    )
    return j_matrix


def main():
    # Initial and final time point.
    t0 = 0
    t_final = 50
    # Initializing the time step and initial guess.
    delta_t = 0.01
    x0 = np.array([1, 1])
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

    # Plotting the behavior of the two variables, x1 and x2.
    plt.plot(iter_array, x1_array, 'r', iter_array, x2_array, 'b')
    # Setting the options for the plots.
    plt.title('Plot of Solution for Implicit Trapezoidal Method')
    plt.xlabel('Iteration #')
    plt.ylabel('Value')
    plt.legend(('x1', 'x2'), loc='upper right')
    plt.savefig('trapezoidal.png')

    # Now we're asked to get the equilibrium points, which occur when the derivative of the functions is 0.
    # Since we made the Newton-Raphson method code, which specializes in looking for roots, we can just use
    # that to calculate the roots. Our first initial guess is just our initial solution.
    root_1 = newton_raphson(x0, f=f_of_x, j=jacobian_of_f)
    # Since we want non-trivial solutions, let's go the negative side of the plane.
    root_2 = newton_raphson(-x0, f=f_of_x, j=jacobian_of_f)
    # Just prints out the roots.
    print(root_1, root_2)

    # Shows the plot after printing out the roots.
    plt.show()


if __name__ == '__main__':
    main()
