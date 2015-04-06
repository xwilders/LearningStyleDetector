from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'Simulator',
  ext_modules = cythonize(["simulatorKurtEfficient.pyx", "learnersKurt.pyx"]),
  include_dirs=[numpy.get_include()]
)