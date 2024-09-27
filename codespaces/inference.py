import subprocess
import sys

# make sure SIMS is installed
try:
    from scsims import SIMS
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "SIMS"])
    from scsims import SIMS  # Retry import after installation
     
from scsims import SIMS
from pytorch_lightning.loggers import WandbLogger
import anndata as an

checkpoint_path = '../checkpoint/myawesomemodel.ckpt'  # Replace 'myawesomemodel.ckpt' with the pretrained model in the checkpoint folder
unlabeled_data_path = '../checkpoint/my_unlabeled_data.h5ad'  # Replace 'my_unlabeled_data.h5ad' with  your testing data

# Load the unlabeled data
unlabeled_data = an.read_h5ad(unlabeled_data_path)

# Initialize the SIMS model
sims = SIMS(weights_path=checkpoint_path)

# Run predictions
cell_predictions = sims.predict(unlabeled_data_path)

# Look at the explainability of the model
explainability_matrix = sims.explain('my/new/unlabeled.h5ad') # this can also be labeled data, of course 

# Save the predictions
cell_predictions.to_csv('predictions.csv')
