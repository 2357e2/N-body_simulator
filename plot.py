import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as Animation
from params import *
from simulation import Simulation

def plot(simulation: Simulation):
    fig = plt.figure(figsize = (8,8)) # Square figure array of length 8 (to fit my laptop screen)
    ax0 = fig.add_axes([0,0,1,1])
    ax1 = ax0.twinx() # Superpose second axis onto first
    ax0.set_xlim(xmin, xmax), ax0.set_xticks([])
    ax0.set_ylim(ymin, ymax), ax0.set_yticks([])
    fig.canvas.manager.window.wm_geometry("+%d+%d" % (10, 10)) # Places window in convenient location on screen
    ax1.set_xlim(xmin, xmax), ax0.set_xticks([])
    ax1.set_ylim(ymin, ymax), ax0.set_yticks([])
    ax2 = ax0.twinx()
    ax2.set_xlim(xmin, xmax), ax0.set_xticks([])
    ax2.set_ylim(ymin, ymax), ax0.set_yticks([])

    simulation.axes = [ax0, ax1, ax2]

    animation = Animation(fig, simulation.update, frames=None, interval=0)
    simulation.animation = animation
    ax0, ax1, ax2 = simulation.axes
    plt.show()
