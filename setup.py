from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("ss_fuzzy.pyx")
)

# setup(
#     ext_modules = cythonize("asd.pyx")
# )