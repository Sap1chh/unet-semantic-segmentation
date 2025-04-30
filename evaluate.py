import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from models.unet import UNet
import numpy as np
import random

# === Параметри ===
DATA_PATH = "./data/small_dataset.pt"
MODEL_PATH = "unet_model.pth"
IMG_SIZE = 128

# === Завантаження датасету ===
data = torch.load(DATA_PATH)
images = data["images"]
masks = data["masks"]

# === Вибір випадкового зразка ===
idx = random.randint(0, len(images) - 1)
image = images[idx].unsqueeze(0)  # додати batch розмір
true_mask = masks[idx].squeeze().numpy()

# === Завантаження моделі ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# === Прогноз ===
with torch.no_grad():
    input_tensor = image.to(device)
    output = model(input_tensor)
    prediction = torch.sigmoid(output).squeeze().cpu().numpy()
    pred_mask = (prediction > 0.5).astype(np.uint8)

# === Візуалізація ===
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
ax1.set_title("🖼️ Original Image")
ax1.imshow(np.transpose(image.squeeze().numpy(), (1, 2, 0)))
ax1.axis("off")

ax2.set_title("✅ Ground Truth Mask")
ax2.imshow(true_mask, cmap="gray")
ax2.axis("off")

ax3.set_title("🔮 Predicted Mask")
ax3.imshow(pred_mask, cmap="gray")
ax3.axis("off")

plt.tight_layout()
plt.show()
