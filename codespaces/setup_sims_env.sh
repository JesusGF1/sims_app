#!/bin/bash

# Create a conda environment named 'simsenv' with Python 3.9
conda create --name simsenv python=3.9 -y

# Activate the new environment
source activate simsenv

# Install SIMS
pip install --use-pep517 git+https://github.com/braingeneers/SIMS.git

# Install the IPython kernel for Jupyter
python -m ipykernel install --user --name simsenv --display-name "simsenv"

pip install --upgrade jupyter ipywidgets

# Instructions for the user
echo "Kernel 'simsenv' has been created. Please select it in your Jupyter notebook."
