from setuptools import Extension, setup
import numpy as np

ext_modules = [
    Extension(
        "matris.graph.cygraph",
        ["matris/graph/cygraph.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]

setup(ext_modules=ext_modules, setup_requires=["Cython"])

# python setup.py build_ext --inplace
