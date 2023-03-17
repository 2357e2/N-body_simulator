# Class declarations
from typing import List
import numpy as np
from params import particle_mass, n_rings
from utils import static_velocity

class Particle:
    def __init__(
        self, 
        particleNumber: int, 
        mass: float, 
        radius: int, 
        angle: float, 
        velocity: List[float], 
        ringNumber: int, 
        galaxyNumber: int
    ) -> None:
        self.particleNumber = particleNumber
        self.mass = mass
        self.radius = radius
        self.angle = angle
        self.velocity = velocity # in polar coords
        self.ringNumber = ringNumber
        self.galaxyNumber = galaxyNumber

class Ring:
    def __init__(self, iden: int, radius: int, particles: List[Particle]) -> None:
        self.id = iden
        self.radius = radius
        self.particles = particles

class Galaxy:
    def __init__(self, mass: float, position: List[float], velocity: List[float]) -> None:
        ringList = []
        for i in range(n_rings):
            particleList = []
            radius = 2+i/2
            for j in range(12*n_rings + 3*n_rings*i): # rings have 10 x (12, 15, 18, 21, 24, 27, 30, 33, 36, 39) particles
                angle = 2*np.pi*j/(12*n_rings + 3*n_rings*i)
                if (angle > np.pi):
                    angle -= 2*np.pi # Angle range -pi to +pi
                particle = Particle(j, particle_mass, radius, angle, static_velocity(radius, mass), i, 0)
                particleList.append(particle)
            ring = Ring(i, radius, particleList) # rings have radii of 2,2.5,3,3.5,4,4.5,5,5.5,6,6.5
            ringList.append(ring)

        self.mass = mass
        self.position = position # in polar coords
        self.velocity = velocity # in polar coords
        self.rings = ringList

class Universe:
    def __init__(self, galaxies: List[Galaxy]) -> None:
        self.galaxies = galaxies