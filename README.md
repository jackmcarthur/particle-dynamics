# particle-dynamics
A small project showing the evolution of a system of gas particles as they cool to temperatures near 0 Kelvin. This model realistically accounts for electrostatic attractive and repulsive forces between atoms, allowing for "crystallization" behavior. 
This project allows one to make animations like the following, which shows 400 particles with a clustering affinity, analagous to a real life situation of immiscible molecules like oil and water:

<img src="particle%20dynamics%20gif.gif" width="400">

<a href="https://i.imgur.com/lZdJoCI.mp4" target="\_blank">Link to a 60 fps .MP4 file of the above gif.</a>

## Cython details
This project is done in Python with some assistance from Cython, a superset of the Python language that allows for static typing, C standard libraries, and other C features to be used in Python. It is used here because it can be immensely faster than Python, especially with computations involving looping over arrays. This function, which is used to update the velocities of an ensemble of particles once per frame by looping over the attractive and repulsive forces between every choice of two particles, is sped up by a factor of nearly 20x:

```
%timeit Ensemble(100, 100, 0., 3., 0., 3.)._velocity_update()
# native python: 78.4 ms ± 1.47 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
# cython: 4.32 ms ± 109 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```
If you have Cython installed, you can download and compile [particledynamics2D.pyx](particledynamics2D.pyx) by executing [setup.py](setup.py) in the command line using `python setup.py build_ext --inplace`. particledynamics2D will then become a normal python module, able to be imported by `import particledynamics2D` like any other.
For an example of how to use or animate an instance of the Ensemble class, see [particledynamicstest.py](particledynamicstest.py), which was used to create the above gif.

The two Jupyter notebooks in this repository can be used with no imports to generate the same results; one is for the 2D case and the other shows an extension of this module to 3D. The 3D case shows the 3-dimensional crystallization of a set of positively and negatively charged ions, and over long simulation times, the particles will settle into real crystal habits such as body-centered cubic.

## Details and construction
The module [particledynamics2D.pyx](particledynamics2D.pyx) contains a Particle class and an Ensemble class. The Ensemble class can be called in the following format:
`Ensemble(n_blue, n_red, x_lower, x_upper, y_lower, y_upper, drag_coeff=0.2, v_in=0.2, C_blue=0.001, C_red=None)`
* `n_blue` is the number of blue particles in the simulation
* `n_red` is the number of red particles in the simulation
* `x_lower` and `x_upper` are the positions of the left and right walls of the bounding box
* `y_lower` and `y_upper` are the positions of the bottom and top walls of the bounding box
* `drag_coeff` is a measure of how fast the particles lose energy, and increasing it will cause faster  crystallization behavior
* `v_in` is the initial velocity of the particles, which can be set anywhere between 0 and 5 as 5 is the maximum allowed speed.
* `C_blue` and `C_red` are the strengths of clustering interactions of blue particles with other blue particles and red particles with other red particles. If not specified, C_red defaults to the same value as C_blue.
Animations like the above can be produced by manipulating [particledynamicstest.py](particledynamicstest.py).
