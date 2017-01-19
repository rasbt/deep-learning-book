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
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a


# Setting up the Test Environment

if [ "${LATEST}" = "true" ]; then
  conda create -q -n testenv --yes python=$TRAVIS_PYTHON_VERSION numpy;
else
  conda create -q -n testenv --yes python=$TRAVIS_PYTHON_VERSION numpy=$NUMPY_VERSION;
fi

source activate testenv

conda install -q -y pip

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"

pip install --upgrade pip

if [ "${COVERAGE}" = "true" ]; then
    pip install coveralls
fi

if [ "${NOTEBOOKS}" = "true" ]; then
    conda install -q -y jupyter matplotlib;
    pip install watermark;
    pip install nbformat;

    # temporary pip install until 0.12 is released on conda
    export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp35-cp35m-linux_x86_64.whl
    if [ "${LATEST}" = "true" ]; then
      # conda install tensorflow;
      pip install $TF_BINARY_URL
    else
      # conda install -q -y tensorflow=$TENSORFLOW_VERSION;
      pip install $TF_BINARY_URL

    fi
fi


python setup.py install;
