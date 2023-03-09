import numpy as np
import scipy
from scipy.integrate import ode
import params

def radiusODE (t, r, angularVelocity, m, M, F):
    """
    Takes input parameter r the dynamic variable. F is the polar vector form of the
    externally applied force. m and M are masses of the small mass and large central mass respectively.
    Returns a list with two elements, the first being d2r/dt2 and the other dr/dt.
    Differential equations are derived from the system's lagrangians.
    dr/dt = a
    da/dt = angularVelocty^2r + GM/r^2 + F.r^/m
    """
    return ((r[1]), (angularVelocity**2*r[0] - params.G*M/r[0]**2 + F/m))

def angleODE (t, angle, r, m, _, F):
    """
    Takes input parameter angle the dynamic variable. F is the polar vector form of the
    externally applied force. r is the radius. m and M are masses of the small mass and 
    large central mass respectively.
    Returns a list with two elements, the first being d2(angle)/dt2 and the other d(angle)/dt.
    Differential equations are derived from the system's lagrangians.
    d(angle)/dt = b
    db/dt = -F.angle^/(mr^2)
    """
    return (angle[1], F/(m*r**2))

def perturbationODE(t, r, _, mp, mg, angularMomentum):
    """
    A = dr/dt
    dA/dt = (GMm^2 - J^2)/m^2
    """
    return (-r[1], (- params.G*mg*mp**2 + angularMomentum**2)/(mp*r[0])**2)

def calcSolution(parameters, crossVar, m, M, F, function):
    """
    Container function for the ODE solver (scipy.integrate.solve_ivp).
    Takes input parameters radius or angle, the dynamic variables. crossVar is the variable of the
    orthogonal direction used in the equation of motion (angularVelocity for radial direction, radius
    for the polar direction). F is the polar vector form of the externally applied force. 
    m and M are masses of the small mass and large central mass respectively.
    function corresponds to the parameter being solved, either radiusODE() or angleODE(). 
    """
    solution = scipy.integrate.solve_ivp(
        fun = function,
        t_span = (0, params.timeIncrement),
        y0=(parameters[0], parameters[1]), # Initial values for DofF and its first derivative
        args=(crossVar, m, M, F), # Additional parameters for ODE function
        t_eval=np.linspace(0, params.timeIncrement, params.samples),
        max_step = 1
    )
    return solution