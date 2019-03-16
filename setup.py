from setuptools import find_packages, setup
from distutils.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext

# References:
#  https://stackoverflow.com/questions/4505747/how-should-i-structure-a-python-package-that-contains-cython-code
#  https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py

# TODO: investigate a cleaner workaround (see SciPy etc.)
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

# Run cython on the pyx files if possible, otherwise rely on the c file
# fastexp.c must be included in the release
ext_modules = []
if use_cython:
    ext = Extension("hiergp.fastexp.fastexp",
                    ["src/hiergp/fastexp/fastexp.pyx",
                     "src/hiergp/fastexp/fexp.c"],
                    include_dirs = ["src/hiergp/fastexp"],
                    extra_link_args=['-lm'],
                    extra_compile_args=['-fomit-frame-pointer',
                                        '-msse2',
                                        '-mfpmath=sse',
                                        '-ffast-math'])
    ext_modules += cythonize(ext)
else:
    ext = Extension("hiergp.fastexp.fastexp",
                    ["src/hiergp/fastexp/fastexp.c",
                     "src/hiergp/fastexp/fexp.c"],
                    include_dirs = ["src/hiergp/fastexp"],
                    extra_link_args=['-lm'],
                    extra_compile_args=['-fomit-frame-pointer',
                                        '-msse2',
                                        '-mfpmath=sse',
                                        '-ffast-math'])
    ext_modules.append(ext)

setup(
    name="hiergp",
    version='0.0.1',
    cmdclass= {'build_ext':build_ext},
    ext_modules=ext_modules,
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    setup_requires=[
        'cython',
        'numpy>=1.8'
    ],
    install_requires=[
        'numpy>=1.8'
    ]
)
