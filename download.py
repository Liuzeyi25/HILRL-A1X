# Load model directly
from transformers import AutoModel
# download into /home/luka/Haoyuan/Safevla_RL/octo_model
model = AutoModel.from_pretrained("rail-berkeley/octo-small-1.5", dtype="auto")  