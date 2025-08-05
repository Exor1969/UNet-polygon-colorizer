import torch
from unet import UNet


#Config
MODEL_PATH = 'polygon_unet_lr_0.001bs64MSEAdamep100.pth'
IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COLOR_MAP = {
    "red": 0, "green": 1, "blue": 2, "yellow": 3, "purple": 4,
    "orange": 5, "cyan": 6, "magenta": 7
}
NUM_COLORS = len(COLOR_MAP)

#Loading the Model
model = UNet(n_channels=1, n_classes=3, num_colors=NUM_COLORS)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
model.to(DEVICE)
model.eval() # Setting the model to evaluation mode

print("Model loaded successfully.")

