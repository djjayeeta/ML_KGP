#from distutils.core import setup
#from Cython.Build import cythonize

#setup(
#    ext_modules = cythonize("ss_fuzzy.pyx")
#)

# setup(
#     ext_modules = cythonize("asd.pyx")
# )
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


ext_modules=[
    Extension("ss_fuzzy",
              ["ss_fuzzy.pyx"],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp']
              )
]

setup(
  name = "ss_fuzzy",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules
)
