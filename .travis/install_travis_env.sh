#!/usr/bin/env bash

# Installing MINICONDA

set -e

if [ "${TRAVIS_PYTHON_VERSION}" == "2.7" ]; then
  wget http://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh;
else
  wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
fi


bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

# Setting up the Test Environment

if [ "${LATEST}" = "true" ]; then
  conda create -q -n testenv --yes python=$TRAVIS_PYTHON_VERSION numpy
else
  conda create -q -n testenv --yes python=$TRAVIS_PYTHON_VERSION numpy=$NUMPY_VERSION
fi

source activate testenv

conda install -q -y pip nose

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"

pip install --upgrade pip

if [ "${COVERAGE}" = "true" ]; then
    pip install coveralls
fi

if [ "${NOTEBOOKS}" = "true" ]; then
    conda install -q -y jupyter matplotlib nbformat
  fi


python setup.py install
