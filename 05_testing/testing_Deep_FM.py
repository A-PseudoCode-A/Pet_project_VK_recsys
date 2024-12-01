import torch
from recbole.utils import get_model, init_seed
from recbole.trainer import Trainer
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction

model_file = "../04_training/DeepFM//saved/DeepFM-Dec-01-2024_21-06-10.pth"
model_class = get_model("DeepFM")
model = model_class.load_model(model_file)
model.eval()
