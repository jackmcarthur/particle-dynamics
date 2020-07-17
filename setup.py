from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("particledynamics2D.pyx"),
    include_dirs=[numpy.get_include()]
)

# type in command line: python setup.py build_ext --inplace
