import particledynamics2D as p2d
import matplotlib.pyplot as plt
from matplotlib import animation

###############################

boxbound = 4.

fig = plt.figure(frameon=False)
fig.set_size_inches(3,3)


ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(0.,boxbound), ylim=(0.,boxbound))
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
fig.subplots_adjust(left=0.02,right=0.98,bottom=0.02,top=0.98)

# blue/redparticles hold the locations of the particles
blueparticles, = ax.plot([], [], 'bo', markersize=2)
redparticles, = ax.plot([], [], 'ro', markersize=2)
en = p2d.Ensemble(70, 30,
              0., boxbound,
              0., boxbound,
              drag_coeff=0.9, v_in=0.4, C_blue=0.0002, C_red=0.0002)

def init():
    bxpoints = en.p_array[:en.nblue,0]
    bypoints = en.p_array[:en.nblue,1]
    blueparticles.set_data(bxpoints, bypoints)
    rxpoints = en.p_array[en.nblue:,0]
    rypoints = en.p_array[en.nblue:,1]
    redparticles.set_data(rxpoints, rypoints)
    return blueparticles, redparticles

def animate(i):
    # this for loop defines how many time steps are skipped between frames
    # change the 4 to speed up or slow down the animation
    for i in range(4):
        en._velocity_update()
        en._time_step()
        en._set_p_array()
    bxpoints = en.p_array[:en.nblue,0]
    bypoints = en.p_array[:en.nblue,1]
    blueparticles.set_data(bxpoints, bypoints)
    rxpoints = en.p_array[en.nblue:,0]
    rypoints = en.p_array[en.nblue:,1]
    redparticles.set_data(rxpoints, rypoints)
    return blueparticles, redparticles

# change number of frames generated to lengthen/shorten animation
anim = animation.FuncAnimation(fig, animate, frames=1200, init_func=init, interval=20)
print("generating...")
# must have FFMpeg installed for this step
anim.save('particles.mp4', dpi=150, fps=60, writer='ffmpeg')
print("finished")
