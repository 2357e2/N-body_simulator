# N-body simulator version 2.0

# Packages
import time
start = time.time()
import matplotlib.pyplot as plot
from matplotlib.animation import FuncAnimation as Animation
import numpy as np
import scipy
from scipy.integrate import ode

# Constants
G = 1

# Simulation parameters
centre = [0,0]
xmin = -18
xmax = 10
ymin = -20
ymax = 8
simulationLength = 24 # update() pause parameter
timeIncrement = 0.5 # ODE solver, t_eval: upper time limit
samples = 10 # ODE solver, t_eval: precision of calculation
perturbationStationary = False
perturbationAngle = -np.pi/2
energy = 0
particleMass = 1e-5


# Independent parameters
rotationDirection = 1
perturbationMass = 0.1
focus = 10
# number of rings
ringTotal = 5

#fileName =  #("rotation direction: " + str(rotationDirection) + "\nperturbation mass: " + str(perturbationMass) + "\nfocus: " + str(focus))
filePosition = open('116Pos.txt', 'w')
filePosition.write("Position\n\n\n")
fileVelocity = open('116Vel.txt', 'w')
fileVelocity.write("Velocity\n\n\n")
filePerturbation = open('116Per.txt', 'w')
filePerturbation.write("Perturbation\n\n\n")

# Class declarations
class Particle:
    def __init__(this, particleNumber, mass, radius, angle, velocity, ringNumber, galaxyNumber):
        this.particleNumber = particleNumber
        this.mass = mass
        this.radius = radius
        this.angle = angle
        this.velocity = velocity # in polar coords
        this.ringNumber = ringNumber
        this.galaxyNumber = galaxyNumber
    
    def destroy(particle):
        """
        'Destroys' particle which enters core of galaxy by setting position to out of range and sets
        velocity equal to zero to prevent further interaction.
        """
        particle.radius = 1e10
        particle.velocity = [0,0]

class Ring:
    def __init__(this, iden, radius, particles):
        this.id = iden
        this.radius = radius
        this.particles = particles

class Galaxy:
    def __init__(this, mass, position, velocity, rings):
        this.mass = mass
        this.position = position # in polar coords
        this.velocity = velocity # in polar coords
        this.rings = rings

class Universe:
    def __init__(this, galaxies):
        this.galaxies = galaxies

# Utility functions
def polar_cartesian(radius, angle, origin):
    """
    Converts position of particle in polar coordinates to cartesian, which it returns
    Takes radius of ring and angle of particle (polar), converts this to
    cartesian and adds vectorially to the galaxy's central position
    """
    relativePosition = [radius*np.cos(angle), radius*np.sin(angle)]
    position = [relativePosition[0] + origin[0], relativePosition[1] + origin[1]]
    return (position)

def cartesian_polar(particlePosition, galaxyCentre):
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

def create_galaxy (position, velocity, mass):
    """
    Creates a galaxy with set parameters and returns it. Set parameters are mass, cartesian position
    and velocity. Requires creating a ringList and a mass.
    The ringList contains rings each of which have a particle list, and this list contains particles.
    """

    # Create ringList
    ringList = []
    for i in range(ringTotal):
        particleList = []
        radius = 2+i/2
        for j in range(12*ringTotal + 3*ringTotal*i): # rings have 10 x (12, 15, 18, 21, 24, 27, 30, 33, 36, 39) particles
            angle = 2*np.pi*j/(12*ringTotal + 3*ringTotal*i)
            if (angle > np.pi):
                angle -= 2*np.pi # Angle range -pi to +pi
            particle = Particle(j, particleMass, radius, angle, static_velocity(radius, mass), i, 0)
            particleList.append(particle)
        ring = Ring(i, radius, particleList) # rings have radii of 2,2.5,3,3.5,4,4.5,5,5.5,6,6.5
        ringList.append(ring)

    galaxy = Galaxy(mass, position, velocity, ringList) # default mass 1, position centre, velocity 0
    return galaxy

def calc_eccentricity(mp, mg):
    """
    Calculates the eccentricity of the perturbation's orbit, determined by its energy.
    """
    C = energy*focus/(G*mg*mp)
    return (C + np.sqrt(1 + 2*C + C**2))

def calc_radius(angle):
    """
    Calculates the radius of the perturbing galaxy's orbit from its angle.
    """
    return(semi/(1+e*np.cos(angle)))

def calc_angle(radius):
    """
    Calculates the angle of orbit from the polar radius. Sign differentiates positive or negative angles
    for the bi-valued radius.
    """
    cosAngle = ((semi/radius-1)/e)
    if (cosAngle > 1):
        return 0
    else:
        return(-np.arccos(cosAngle))

def radiusODE (t, r, angularVelocity, m, M, F):
    """
    Takes input parameter r the dynamic variable. F is the polar vector form of the
    externally applied force. m and M are masses of the small mass and large central mass respectively.
    Returns a list with two elements, the first being d2r/dt2 and the other dr/dt.
    Differential equations are derived from the system's lagrangians.
    dr/dt = a
    da/dt = angularVelocty^2r + GM/r^2 + F.r^/m
    """
    return ((r[1]), (angularVelocity**2*r[0] - G*M/r[0]**2 + F/m))

def angleODE (t, angle, r, m, M, F):
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

def perturbationODE(t, r, crossVar, mp, mg, angularMomentum):
    """
    A = dr/dt
    dA/dt = (GMm^2 - J^2)/m^2
    """
    return (-r[1], (- G*mg*mp**2 + angularMomentum**2)/(mp*r[0])**2)

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
        t_span = (0, timeIncrement),
        y0=(parameters[0], parameters[1]), # Initial values for DofF and its first derivative
        args=(crossVar, m, M, F), # Additional parameters for ODE function
        t_eval=np.linspace(0, timeIncrement, samples),
        max_step = 1
    )
    return solution

def static_velocity (radius, mass):
    """
    Works out the angular velocity of a particle for the static, unperturbed galaxy.
    Uses mv**2/r = GMm/r**2. Returns angular velocity required for this to balance. Central mass input.
    Used to set the initial angular velocity
    """
    angularVelocity = rotationDirection*np.sqrt(G*mass/radius**3) # rotationDirection determines the sign of angularVelocity
    return [0, angularVelocity] # 0 radial velocity, angularVelocity in theta direction.

def calc_force (thisGalaxy, otherGalaxy, particleNumber, ringNumber):
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


# Fixed variable declarations
e = calc_eccentricity(perturbationMass, 1) # eccentricity
semi = focus*(1+e) # semi latus rectum 
angularMomentum = np.sqrt(G*perturbationMass**2*1*semi)


# Modelling functions
def setup():
    """
    Sets up simulation by creating two galaxies and calculates angular momentum/mass of perturbing galaxy,
    """
    galaxy = create_galaxy(centre, [0,0], 1)
    radius = calc_radius(perturbationAngle)
    angularVelocity = angularMomentum/(perturbationMass*radius**2)
    radialVelocity = np.sqrt(2*energy/perturbationMass - radius**2*angularVelocity**2 + 2*G*1/radius)
    perturbation = create_galaxy([radius, perturbationAngle], [radialVelocity, angularVelocity], perturbationMass)

    universe = Universe([galaxy, perturbation])
    return galaxy, perturbation, universe

galaxy, perturbation, universe = setup()

def update (timeVal):
    """
    Updates animation with set time interval. Increments each particle's angular position by angular velocity
    No returns. If time exceeds simulationLength, stops.
    """
    perturbationPosition = polar_cartesian(perturbation.position[0], perturbation.position[1], centre)
    galaxyPosition = polar_cartesian(galaxy.position[0], galaxy.position[1], centre)

    # Plot positions of galaxies' particles
    scatterPointsx = []
    scatterPointsy = []
    positions = []
    velocities = []
    for ring in galaxy.rings:
        for particle in ring.particles:
            angle = particle.angle
            radius = particle.radius
            positions.append([radius, angle]) # position data takes form of [radius, angle]
            velocities.append(particle.velocity)
            position = polar_cartesian(radius, angle, centre)
            scatterPointsx.append(position[0])
            scatterPointsy.append(position[1])

    scatter = ax.scatter(scatterPointsx, scatterPointsy, s=10, lw=0.5, edgecolors='black', facecolors='black')
    scatter1 = ax1.scatter(perturbationPosition[0], perturbationPosition[1], s=500, lw=0.5, edgecolors='red', facecolors='red')
    scatter2 = ax2.scatter(galaxyPosition[0], galaxyPosition[1], s=500, lw=0.5, edgecolors='black', facecolors='black')
    
    # Enforce simulationLength
    if (timeVal > simulationLength):
        animation.pause()
        print(time.time()-start)

    # Update particle's positions and velocities
    radiusList = []
    velocityList = []
    angleList = []
    for ring in galaxy.rings:
        for particle in ring.particles:
            force = calc_force(galaxy, perturbation, particle.particleNumber, particle.ringNumber)
            solutionR = calcSolution([particle.radius, particle.velocity[0]], particle.velocity[1], particle.mass, galaxy.mass, force[0], radiusODE)
            solutionT = calcSolution([particle.angle, particle.velocity[1]], particle.radius, particle.mass, galaxy.mass, force[1], angleODE)
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
    mp = perturbation.mass
    mg = galaxy.mass
    if (not perturbationStationary):
        radius, angle = perturbation.position
        radialVelocity, angularVelocity = perturbation.velocity
        solution = calcSolution([radius, radialVelocity], None, mp, mg, angularMomentum, perturbationODE)
        radius = solution.y[0][-1]
        radialVelocity = solution.y[1][-1]
        angle = calc_angle(radius)
        if (angle == 0):
            animation.pause() # Pauses animation once angle exceeds 0
            print(time.time() - start)
        angularVelocity = angularMomentum/(mp*radius**2)
        perturbation.position = [radius, angle]
        perturbation.velocity = [radialVelocity, angularVelocity]

    # Store data
    for i in range(len(radiusList)):
        filePosition.write(str(radiusList[i]) + "\n")
        filePerturbation.write(str(angleList[i]) + "\n")
        #fileVelocity.write(str(velocityList[i]) + "\n")


# Plot and Animation creation
fig = plot.figure(figsize = (8,8)) # Square figure array of length 8 (to fit my laptop screen)
ax = fig.add_axes([0,0,1,1])
ax1 = ax.twinx() # Superpose second axis onto first
ax.set_xlim(xmin, xmax), ax.set_xticks([])
ax.set_ylim(ymin, ymax), ax.set_yticks([])
fig.canvas.manager.window.wm_geometry("+%d+%d" % (10, 10)) # Places window in convenient location on screen
ax1.set_xlim(xmin, xmax), ax.set_xticks([])
ax1.set_ylim(ymin, ymax), ax.set_yticks([])
ax2 = ax.twinx()
ax2.set_xlim(xmin, xmax), ax.set_xticks([])
ax2.set_ylim(ymin, ymax), ax.set_yticks([])

animation = Animation(fig, update, frames=None, interval=0)
plot.show()

filePerturbation.close()
filePosition.close()
fileVelocity.close()