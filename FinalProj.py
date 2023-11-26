import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

#all variables follow meter, kilogram, seconds (MKS) system


    
def non_dim_solver(time, state):
    '''Takes the intial position and velocity of the third mass (negligible one) and the mass ratio (pi2).'''
    position = state[0:2]
    velocity = state[2:4]
    pi2 = state[4]

    pi1 = 1.0-pi2
    sigma = np.sqrt((position[0] + pi2)**2 + position[1]**2) # nondimensional distance to mass 2
    psi = np.sqrt((position[0] - pi1)**2 + position[1]**2) # nondimensional distance to mass 1

    acceleration = np.zeros(2)
    acceleration[0] = 2*velocity[1] + position[0] - pi1*(position[0]+pi2) / sigma**3 - pi2*(position[0]-pi1) / psi**3 
    acceleration[1] = -2*velocity[0] + position[1] - pi1*position[1]/sigma**3 - pi2*position[1]/psi**3

    return np.array([velocity[0], velocity[1], acceleration[0], acceleration[1], 0.0]) # derivatives for scipy to solve system of DEs


def earth_moon_system(t):
    '''Numerically solves the system for the Earth-Moon system. Does not take into account the sun'''
    mass_earth = 5.974e24
    mass_moon = 7.348e22
    total_mass = mass_earth + mass_moon

    # use the same dimensionless variables as Weber
    pi1 = mass_earth/total_mass
    pi2 = mass_moon/total_mass

    initial_state = np.array([pi1*np.cos(t), pi1*np.sin(t), 0.0, 0.0, pi2])

    time_points = np.arange(0, 40.0, 0.01) # rather arbitrary
    solution = solve_ivp(non_dim_solver, [time_points[0], time_points[-1]], initial_state, t_eval=time_points)

    xvals_moon = pi1 * np.cos(np.linspace(0.0, np.pi*2, 1000))
    yvals_moon = pi1 * np.sin(np.linspace(0.0, np.pi*2, 1000))

    plt.title('Earth-Moon trajectory (L4)')
    plt.plot(xvals_moon, yvals_moon, 'k--', zorder=0)
    plt.plot(solution.y.T[:,0], solution.y.T[:,1], label='Trajectory', zorder=0)
    plt.scatter(-pi2, 0.0, color='y', s=120, label='Earth', zorder=10)
    plt.scatter(pi1, 0.0, color='k', s=40, label='Moon')
    plt.xlim([-2.0, 2.0])
    plt.ylim([-2.0, 2.0])
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()

    return solution

earth_moon_system(1/3 * np.pi)





