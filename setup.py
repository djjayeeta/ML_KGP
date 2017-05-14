from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("fuzzy_c_spatial_spectral.pyx")
)

# setup(
#     ext_modules = cythonize("asd.pyx")
# )