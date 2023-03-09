from params import *
from utils import calc_eccentricity
from simulation import Simulation
from plot import plot
import time

if __name__ == '__main__':
    start = time.time()

    simulation = Simulation(start, record_data)

    if record_data:
        filePosition = open('116Pos.txt', 'w')
        filePosition.write("Position\n\n\n")
        fileVelocity = open('116Vel.txt', 'w')
        fileVelocity.write("Velocity\n\n\n")
        filePerturbation = open('116Per.txt', 'w')
        filePerturbation.write("Perturbation\n\n\n")
        simulation.filePosition = filePosition
        simulation.fileVelocity = fileVelocity
        simulation.filePerturbation = filePerturbation
    
    plot(simulation)
    
    if record_data:
        filePerturbation.close()
        filePosition.close()
        fileVelocity.close()