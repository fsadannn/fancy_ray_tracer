import os
import re
from distutils.core import setup

import numpy
from Cython.Build import cythonize


def build_ext(path: str, use_numpy=True):
    exts = cythonize(path,
                     force=True, compiler_directives={'language_level': '3'})
    if use_numpy:
        for ext in exts:
            ext.include_dirs = [numpy.get_include()]
    return exts


#extension = build_ext('fancy_ray_tracer/compiled/*.pyx')
extension = build_ext('fancy_ray_tracer/compiled/_intersection.pyx')
# define_macros=[("CYTHON_TRACE_NOGIL", "1")]

setup(ext_modules=extension, verbose=3,)


def purge(path, pattern):
    for f in os.listdir(path):
        filename = re.search(pattern, f)
        if filename is None:
            pass
        else:
            os.remove(os.path.join(path, f))


purge(os.path.join(os.path.curdir, 'fancy_ray_tracer', 'compiled'), r'.*\.c$')
