# Parameter declarations
import numpy as np

G = 1
n_rings = 5
particle_mass = 1e-5
energy = 0
centre = [0,0]
xmin = -7
xmax = 7
ymin = -8
ymax = 6
simulationLength = 24 # update() pause parameter
timeIncrement = 0.5 # ODE solver, t_eval: upper time limit
samples = 10 # ODE solver, t_eval: precision of calculation
perturbationStationary = False
perturbationAngle = -np.pi/2
rotationDirection = 1
perturbationMass = 0.1
focus = 5

record_data = False