#!/usr/bin/env bash

# Installing MINICONDA

set -e

if [ "${PYTHON_VERSION}" == "2.7" ]; then
  wget http://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh;
else
  wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
fi


bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

# Setting up the Test Environment


conda create -q -n testenv --yes python=$TRAVIS_PYTHON_VERSION numpy;


source activate testenv

conda install pip nose

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"

pip install --upgrade pip

if ["${COVERAGE}" = "true" ]; then
    pip install coveralls
fi

python setup.py install
