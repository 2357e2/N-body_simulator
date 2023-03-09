# Utility functions
import numpy as np
from params import *
from typing import List

def polar_cartesian(radius: float, angle: float, origin: List[float]) -> List[float]:
    """
    Converts position of particle in polar coordinates to cartesian, which it returns
    Takes radius of ring and angle of particle (polar), converts this to
    cartesian and adds vectorially to the galaxy's central position
    """
    relativePosition = [radius*np.cos(angle), radius*np.sin(angle)]
    position = [relativePosition[0] + origin[0], relativePosition[1] + origin[1]]
    return (position)

def cartesian_polar(particlePosition: List[float], galaxyCentre: List[float]) -> List[float]:
    """
    Converts position of particle from cartesian coordinates to polar (with centre defined
    as centre of galaxy) which it returns. Theta postive if in top half of galaxy.
    x, y relative position found from galaxy centre and converts to radius of ring and angle of particle (polar).
    """
    relativePosition = [particlePosition[0] - galaxyCentre[0], particlePosition[1] - galaxyCentre[1]]
    if (relativePosition[0] == 0):
        return([np.sqrt(relativePosition[0]**2 + relativePosition[1]**2), np.sign(relativePosition[1])*np.pi/2])
    if (relativePosition[0] < 0 and relativePosition[1] < 0):
        angle = -np.pi + abs(np.arctan(relativePosition[1]/relativePosition[0]))
    elif(relativePosition[0] > 0 and relativePosition[1] < 0):
        angle = - abs(np.arctan(relativePosition[1]/relativePosition[0]))
    elif(relativePosition[0] < 0 and relativePosition[1] > 0):
        angle = np.pi - abs(np.arctan(relativePosition[1]/relativePosition[0]))
    else:
        angle = abs(np.arctan(relativePosition[1]/relativePosition[0]))
    return ([np.sqrt(relativePosition[0]**2 + relativePosition[1]**2), angle])

def static_velocity (radius: float, mass: float) -> List[float]:
    """
    Works out the angular velocity of a particle for the static, unperturbed galaxy.
    Uses mv**2/r = GMm/r**2. Returns angular velocity required for this to balance. Central mass input.
    Used to set the initial angular velocity
    """
    angularVelocity = rotationDirection*np.sqrt(G*mass/radius**3) # rotationDirection determines the sign of angularVelocity
    return [0, angularVelocity] # 0 radial velocity, angularVelocity in theta direction.

def calc_eccentricity(mp: float, mg: float) -> float:
    """
    Calculates the eccentricity of the perturbation's orbit, determined by its energy.
    """
    C = energy*focus/(G*mg*mp)
    return (C + np.sqrt(1 + 2*C + C**2))

def calc_radius(angle: float, semi: float, e: float) -> float:
    """
    Calculates the radius of the perturbing galaxy's orbit from its angle.
    """
    return(semi/(1+e*np.cos(angle)))

def calc_angle(radius: float, semi: float, e: float) -> float:
    """
    Calculates the angle of orbit from the polar radius. Sign differentiates positive or negative angles
    for the bi-valued radius.
    """
    cosAngle = ((semi/radius-1)/e)
    if (cosAngle > 1):
        return 0
    else:
        return(-np.arccos(cosAngle))

def calc_force (thisGalaxy, otherGalaxy, particleNumber: int, ringNumber: int) -> List[float]:
    """
    Calculates force between given particle and the perturbing galaxy's central mass
    requires knowing the galaxy and which particle is in question.
    Value of G is set, distance and mass scales are in arbitrary units.
    Returns force calculated by gravitational point source formula.
    Resolved into components parallel to radial and polar particle vectors.
    """
    ring = thisGalaxy.rings[ringNumber]
    particle = ring.particles[particleNumber]
    origin = polar_cartesian(thisGalaxy.position[0], thisGalaxy.position[1], centre)
    particlePosition = polar_cartesian(particle.radius, particle.angle, origin)
    otherPosition = polar_cartesian(otherGalaxy.position[0], otherGalaxy.position[1], centre) # Cartesian position of other galaxy
    deltaX = otherPosition[0] - particlePosition[0]
    deltaY = otherPosition[1] - particlePosition[1]
    distance = np.sqrt(deltaX**2 + deltaY**2)
    angleForce = np.arccos(-deltaX/distance)
    angleParticle = particle.angle
    force = G*otherGalaxy.mass*particle.mass/distance**2
    radialForce = -force*(np.cos(angleForce)*np.cos(angleParticle) + np.sin(angleForce)*np.sin(angleParticle))
    angularForce = -force*rotationDirection*(-np.cos(angleForce)*np.sin(angleParticle) + np.sin(angleForce)*np.cos(angleParticle))
    return [radialForce, angularForce]