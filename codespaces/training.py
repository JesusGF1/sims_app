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

logger = WandbLogger(offline=True)

train_data = an.read_h5ad('my_labeled_data.h5ad')
sims = SIMS(data=train_data, class_label='class_label')
sims.setup_trainer(accelerator="gpu", devices=1, logger=logger)
sims.train()