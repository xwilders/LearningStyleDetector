from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'Simulator',
  ext_modules = cythonize(["simulatorKurtEfficient.pyx", "learnersKurt.pyx"]),
)