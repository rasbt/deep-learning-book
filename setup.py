import ann
from setuptools import setup, find_packages

VERSION = ann.__version__

setup(
    name="ann",
    version=VERSION,
    packages=find_packages(),
    install_requires=['numpy>=1.10.4'],
    extras_require={'testing': ['nose']},
    author="Sebastian Raschka",
    author_email="mail@sebastianraschka.com",
    description=("Supporting package for the book "
                 "'Introduction to Artificial Neural Networks "
                 "and Deep Learning: "
                 "A Practical Guide with Applications in Python'"),
    license="MIT",
    keywords=["artificial neural networks", "deep learning",
              "machine learning", "artificial intelligence", "data science"],
    classifiers=[
         'License :: OSI Approved :: MIT License',
         'Operating System :: Microsoft :: Windows',
         'Operating System :: POSIX',
         'Operating System :: Unix',
         'Operating System :: MacOS',
         'Programming Language :: Python :: 2.7',
         'Programming Language :: Python :: 3.5',
         'Programming Language :: Python :: 3.6',
         'Topic :: Scientific/Engineering',
         'Topic :: Scientific/Engineering :: Artificial Intelligence',
         'Topic :: Scientific/Engineering :: Information Analysis',
         'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    url="https://github.com/rasbt/deep-learning-book")
