#!/usr/bin/env bash

# Installing MINICONDA

set -e


DOWNLOAD_PATH="miniconda.sh"

if [ ${PYTHON_VERSION} == "2.7" ]; then
  wget http://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O ${DOWNLOAD_PATH}
  INSTALL_FOLDER="$HOME/miniconda2"
else
  wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${DOWNLOAD_PATH}
  INSTALL_FOLDER="$HOME/miniconda3"
fi


echo "Installing miniconda for python ${PYTHON_VERSION} to ${INSTALL_FOLDER}"
bash ${DOWNLOAD_PATH} -b -p ${INSTALL_FOLDER}

rm ${DOWNLOAD_PATH}

export PATH="$INSTALL_FOLDER/bin:$PATH"

conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a


# Setting up the Test Environment


conda create -n testenv --yes pip python=$PYTHON_VERSION
source activate testenv

echo "Setting up the test environment for python $PYTHON_VERSION"

if [${LATEST} = "true"]; then
    conda install --yes -q numpy nose
else
    conda install --yes -q numpy=${NUMPY_VERSION} nose
fi


python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"

pip install --upgrade pip

if [${COVERAGE} = "true" ]; then
    pip install coveralls
fi

python setup.py install
