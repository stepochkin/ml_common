#!/usr/bin/env python3

from setuptools import setup, find_packages
# from distutils.extension import Extension
# import subprocess
# subprocess.check_call('pip install numpy Cython', shell=True)
# import numpy
# from Cython.Build import cythonize

test_deps = [
    "nose2",
    'nose',
    "mock",
]

setup(
    name="stepin_common_lib",
    version="0.2",
    author="Stepochkin Alexander",
    author_email="astepochkin@stch.ru",
    description="Stepin Common Library",
    url='file://./dist',
    install_requires=[
        'numpy', 'scipy',
        'zstandard',
        # 'psycopg2-binary',
    ],
    package_dir={'': 'lib'},
    packages=find_packages(
        'lib',
        exclude=['*.test']
    ),
    scripts=[],
    tests_require=test_deps,
    test_suite='nose2.collector.collector',
    extras_require={
        'test': test_deps,
    },
    # ext_modules=cythonize([
    #     Extension(
    #         "*",
    #         ["lib/stepin/ml/*.pyx"],
    #         language="c++",
    #         extra_compile_args=["-std=c++11"],
    #         extra_link_args=["-std=c++11"],
    #     ),
    # ], language_level=sys.version_info[0]),
    # include_dirs=[numpy.get_include()],
    zip_safe=True,
)
