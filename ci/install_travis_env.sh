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
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a


# Setting up the Test Environment

if ["${LATEST}" = "true"]; then
    create -q -n testenv --yes -python=$PYTHON_VERSION numpy
else
    create -q -n testenv --yes -python=$PYTHON_VERSION numpy=$NUMPY_VERSION
fi

source activate testenv

conda install pip nose

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"

pip install --upgrade pip

if ["${COVERAGE}" = "true" ]; then
    pip install coveralls
fi

python setup.py install
