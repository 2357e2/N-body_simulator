# Simulation
import time
from objects import Universe, Galaxy
from utils import *
from params import *
from solver import (
    calcSolution, 
    radiusODE, 
    angleODE, 
    perturbationODE
)

class Simulation:

    def __init__(self, start: float, record_data: bool = True) -> None:
        """
        Sets up simulation by creating two galaxies and calculates angular momentum/mass of perturbing galaxy,
        """
        self.e = calc_eccentricity(perturbationMass, 1) # eccentricity
        self.semi = focus*(1+self.e) # semi latus rectum 
        self.angular_momentum = np.sqrt(G*perturbationMass**2*1*self.semi)

        radius = calc_radius(perturbationAngle, self.semi, self.e)
        angularVelocity = self.angular_momentum/(perturbationMass*radius**2)
        radialVelocity = np.sqrt(2*energy/perturbationMass - radius**2*angularVelocity**2 + 2*G*1/radius)

        self.galaxy = Galaxy(1, centre, [0,0])
        self.perturbation = Galaxy(
            perturbationMass, [radius, perturbationAngle], [radialVelocity, angularVelocity]
        )
        self.universe = Universe([self.galaxy, self.perturbation])
        self.start = start
        self.record_data = record_data

    def update (self, timeVal):
        """
        Updates animation with set time interval. Increments each particle's angular position by angular velocity. 
        No returns. If time exceeds simulationLength, stops.
        """
        perturbationPosition = polar_cartesian(self.perturbation.position[0], self.perturbation.position[1], centre)
        galaxyPosition = polar_cartesian(self.galaxy.position[0], self.galaxy.position[1], centre)

        # Plot positions of galaxies' particles
        scatterPointsx = []
        scatterPointsy = []
        positions = []
        velocities = []
        for ring in self.galaxy.rings:
            for particle in ring.particles:
                angle = particle.angle
                radius = particle.radius
                positions.append([radius, angle]) # position data takes form of [radius, angle]
                velocities.append(particle.velocity)
                position = polar_cartesian(radius, angle, centre)
                scatterPointsx.append(position[0])
                scatterPointsy.append(position[1])

        ax0, ax1, ax2 = self.axes

        scatter = ax0.scatter(scatterPointsx, scatterPointsy, s=10, lw=0.5, edgecolors='black', facecolors='black')
        scatter1 = ax1.scatter(perturbationPosition[0], perturbationPosition[1], s=500, lw=0.5, edgecolors='red', facecolors='red')
        scatter2 = ax2.scatter(galaxyPosition[0], galaxyPosition[1], s=500, lw=0.5, edgecolors='black', facecolors='black')
        
        # Enforce simulationLength
        if (timeVal > simulationLength):
            self.animation.pause()
            print(f"Simulation length: {time.time()-self.start}s")

        # Update particle's positions and velocities
        radiusList = []
        velocityList = []
        angleList = []
        for ring in self.galaxy.rings:
            for particle in ring.particles:
                force = calc_force(self.galaxy, self.perturbation, particle.particleNumber, particle.ringNumber)
                solutionR = calcSolution([particle.radius, particle.velocity[0]], particle.velocity[1], particle.mass, self.galaxy.mass, force[0], radiusODE)
                solutionT = calcSolution([particle.angle, particle.velocity[1]], particle.radius, particle.mass, self.galaxy.mass, force[1], angleODE)
                radius, velocityR = solutionR.y[0], solutionR.y[1]
                angle, velocityT = solutionT.y[0], solutionT.y[1]
                # Take the final value from each list for most recent parameter
                particle.radius = radius[-1]
                if (particle.ringNumber == 0):
                    radiusList.append(radius[-1])
                    angleList.append(angle[-1])
                particle.angle = angle[-1]
                particle.velocity = [velocityR[-1], velocityT[-1]]
        
        # Update position of perturbing galaxy
        mp = self.perturbation.mass
        mg = self.galaxy.mass
        if (not perturbationStationary):
            radius, angle = self.perturbation.position
            radialVelocity, angularVelocity = self.perturbation.velocity
            solution = calcSolution([radius, radialVelocity], None, mp, mg, self.angular_momentum, perturbationODE)
            radius = solution.y[0][-1]
            radialVelocity = solution.y[1][-1]
            angle = calc_angle(radius, self.semi, self.e)
            if (angle == 0):
                self.animation.pause() # Pauses animation once angle exceeds 0
                print(time.time() - self.start)
            angularVelocity = self.angular_momentum/(mp*radius**2)
            self.perturbation.position = [radius, angle]
            self.perturbation.velocity = [radialVelocity, angularVelocity]

        # Store data
        if self.record_data:
            for i in range(len(radiusList)):
                self.filePosition.write(str(radiusList[i]) + "\n")
                self.filePerturbation.write(str(angleList[i]) + "\n")
                self.fileVelocity.write(str(velocityList[i]) + "\n")