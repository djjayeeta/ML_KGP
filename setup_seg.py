# from distutils.core import setup
# from Cython.Build import cythonize

# setup(
#    ext_modules = cythonize("mrf_new_simaneal.pyx")
# )

# setup(
#     ext_modules = cythonize("ss_fuzzy.pyx")
# )

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


ext_modules=[
    Extension("mrf_new_simaneal",
              ["mrf_new_simaneal.pyx"],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp']
              )
]

setup(
  name = "mrf_new_simaneal",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules
)
