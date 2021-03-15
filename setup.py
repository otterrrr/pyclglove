try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

long_description = """\
This is another python version of stanford GloVe word embeddings (https://github.com/stanfordnlp/GloVe) parallelized by OpenCL
Most of this work was to port pure python code(https://github.com/otterrrr/pyglove) written into OpenCL-based one so it has almost same logic as that of pure python GloVe implementation
However for simplicity, this pyclglove doesn't concern of memory-aware execution so your system should have enough memory in order to load your own corpus and intermediate results
For your information, GPU-based acceleration becomes available in this version thanks to pyopencl python module thus you can take advantage of any devices supporting OpenCL
"""

setup(
    name='pyclglove',
    version='0.1.0',
    description=('python and OpenCL implementation of GloVe word embeddings'),
    long_description='',
    py_modules=['pyclglove'],
    install_requires=['numpy', 'pyopencl'],
    author='Taesik Yoon',
    author_email='taesik.yoon.02@gmail.com',
    url='https://github.com/otterrrr/pyclglove',
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT license",
        "Topic :: Scientific/Engineering :: Natural Language Processing",
        "Operating System :: OS Independent"
    ]
)
