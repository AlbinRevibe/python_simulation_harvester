import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.optimize import minimize
def simulation(t, x_0, param):
    print('***')
    print(param)
    print('***')
    lambda_m = param[0]
    lambda_e = param[1]
    omega_d = param[2]
    m = param[3]

    omega_0 = np.sqrt((lambda_e+lambda_m)**2 + omega_d**2)
    xi_m = lambda_m/omega_0
    xi_e = lambda_e/omega_0
    xi = xi_e + xi_m


    k = omega_0**2*m
    R_coil = 300
    R_load = 1000
    L = 0.01
    k_t = np.sqrt(lambda_e*(R_coil + R_load)*m)


    args = (omega_0, xi, k_t, R_coil+R_load, L)

    x = odeint(calc_derivatives, x_0, t, args=args)

    pos = x[:, 0]
    vel = x[:, 1]
    current = x[:, 2]
    emf = k_t*vel


    E_m = cumtrapz(calc_mechanical_dissipation(emf, t, k_t, xi_m, omega_0), t)
    E_c = cumtrapz(calc_coil_dissipation(current, R_coil), t)
    E_L = cumtrapz(calc_load_dissipation(current, R_load), t)

    E_kin = m*vel**2/2
    E_pot = k*pos**2/2


    '''
    fig_motion = plt.figure()
    ax_pos = fig_motion.add_subplot(311)
    ax_pos.plot(t, pos)
    ax_vel = fig_motion.add_subplot(312)
    ax_vel.plot(t, vel)
    ax_current = fig_motion.add_subplot(313)
    ax_current.plot(t, current)


    '''
    return(pos, vel, current, emf, E_pot, E_kin, E_m, E_c, E_L)


def calc_derivatives(x, t, omega_0, xi, k_t, R, L):
    A = np.array([[0, 1, 0],
                  [-omega_0**2, -2*omega_0*xi, 0],
                  [0, -k_t/L, -R/L]])

    return np.matmul(A,x)

def calc_nonlinnear_spring():
    print('asda')

def calc_mechanical_dissipation(emf, t, k_t, xi_m, omega_0):
    return (xi_m*omega_0/k_t**2)*emf**2

def calc_coil_dissipation(current, R_coil):
    return current**2/R_coil

def calc_load_dissipation(current, R_load):
    return current**2/R_load

def iterate_mass(m, args):
    t = args[0]
    x_0 = args[1]
    param = args[2]
    param = np.array([param[0], param[1], param[2], m])

    (pos, vel, current, emf, E_pot, E_kin, E_m, E_c, E_L) = simulation(t, x_0, param)

    E_diss = E_m + E_c + E_L
    E_diff = E_pot[0] - E_diss[-1]

    return E_diff**2

if __name__ == '__main__':
    '''
    This file contains a self adjusting loop which calculates the mass given
    the damped frequency and the damping factor lambda.
    '''

    ''' Meassured parameters '''
    with open('damping_model_D.txt', 'r') as textfile:
        next(textfile)
        lambda_m = float(next(textfile))
        lambda_e = float(next(textfile))
        omega_d = float(next(textfile))
    param = [lambda_m, lambda_e, omega_d]

    ''' Simulation parameters '''
    t_0 = 0
    t_f = 0.5
    dt = 0.0001
    t = np.arange(t_0, t_f, dt)

    ''' Initial conditions '''
    pos_0 = 1
    vel_0 = 0
    current_0 = 0
    x_0 = np.array([pos_0, vel_0, current_0])

    m_0 = 0.0800108 # Mass, this is not a free variable! I just did not manage to solve it yet
    args = [t, x_0, param]
    res = minimize(iterate_mass, m_0, args=args)
    m = res.x[0]

    param.append(m)
    (pos, vel, current, emf, E_pot, E_kin, E_m, E_c, E_L) = simulation(t, x_0, param)

    E_diss = E_m + E_c + E_L
    E_tot = E_diss + E_kin[1:] + E_pot[1:]
    E_internal = E_kin[1:] + E_pot[1:]


    fig_energy = plt.figure()
    ax_energy = fig_energy.add_subplot(111)
    ax_energy.plot(t[1:], E_diss, label='Dissipated energy', color='red')
    ax_energy.plot(t[1:], E_internal, label='Energy in spring', color='blue')
    ax_energy.plot(t[1:], E_tot, label='Total energy', color='black', linestyle='--')

    ax_energy.legend()


    plt.show()
