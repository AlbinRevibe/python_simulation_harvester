import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
def simulation():
    xi = 0.01
    omega_0 = 1
    k_t = 1
    args = (omega_0, xi)


    x_0 = np.array([1, 1])
    t_0 = 0
    t_f = 500
    dt = 0.001
    t = np.arange(t_0, t_f, dt)

    x = odeint(calc_derivatives, x_0, t, args=args)

    plt.plot(x[:, 1])
    plt.show()


def calc_derivatives(x, t, omega_0, xi):
    A = np.array([[0, 1],[-omega_0**2, -2*omega_0*xi]])
    return np.matmul(A,x)


if __name__ == '__main__':
    simulation()
