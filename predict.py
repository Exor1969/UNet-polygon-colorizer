import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from load import IMG_SIZE
from load import COLOR_MAP
from load import model
from load import DEVICE


def predict(model, image_path, color_name, device):
    """Generates a colored polygon from an input image and color name."""

    # Preprocess Input Image
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert("L")
    input_tensor = transform(img).unsqueeze(0).to(device) # Add batch dimension

    # Preprocess Color
    if color_name not in COLOR_MAP:
        raise ValueError(f"Color '{color_name}' is not supported. Available colors: {list(COLOR_MAP.keys())}")

    color_idx = COLOR_MAP[color_name]
    color_tensor = torch.tensor([[color_idx]], dtype=torch.long).to(device) # Add batch dimension

    #  Getting Prediction
    with torch.no_grad():
        output_tensor = model(input_tensor, color_tensor)

    #  Post-process for visualization
    output_tensor = output_tensor.squeeze(0).cpu().permute(1, 2, 0) # H, W, C
    output_image = (output_tensor.numpy() * 255).astype(np.uint8)

    return Image.fromarray(output_image)


polygon = input("Enter polygon name: ")
color = input("Enter color name: ")

INPUT_IMAGE_PATH = f'dataset/training/inputs/{polygon}.png'
INPUT_COLOR = f'{color}'

# Generate and Visualize
try:
    predicted_image = predict(model, INPUT_IMAGE_PATH, INPUT_COLOR, DEVICE)

    # Display results
    original_image = Image.open(INPUT_IMAGE_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title(f"Input Image: '{INPUT_IMAGE_PATH}'")
    axes[0].axis('off')

    axes[1].imshow(predicted_image)
    axes[1].set_title(f"Predicted Output (Color: {INPUT_COLOR})")
    axes[1].axis('off')

    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")