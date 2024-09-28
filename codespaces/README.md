# `codespaces folder`

## This folder is meant for users running SIMS via GitHub Codespaces. 

#### This folder contains:
- A bash script for setting up the SIMS virtual environment.
- A Jupyter Notebook to be used in GitHub Codespaces for running SIMS.

#### Once you have created a codespace with this repository, you should drag and drop your `.h5ad` file(s) here so they can be used for training and inference in the Jupyter Notebook.

#### Activate your SIMS virtual environment by running the bash script with these commands:
```chmod +x setup_sims_env.sh```

```./setup_sims_env.sh```

#### Open the notebook, choose the `sims_env` as the kernel, and follow the instructions to begin training and/or inference.

#### Things that may end up in this folder (within your workspace):
- If you perform training, your `lightning_logs` directory with automatically saved model checkpoints will appear within this folder.
- If you perform predictions, your `.csv` file containing those predictions will appear in this folder.