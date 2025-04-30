import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from models.unet import UNet
from tqdm import tqdm
import numpy as np

# === Параметри ===
DATA_PATH = "./data/small_dataset.pt"
BATCH_SIZE = 4
IMG_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3

# === Перевірка наявності збереженого датасету ===
if os.path.exists(DATA_PATH):
    print("✅ Завантажуємо наявний датасет...")
    data = torch.load(DATA_PATH)
    images, masks = data["images"], data["masks"]
else:
    print("📦 Генеруємо новий малий датасет...")
    num_samples = 100
    images = []
    masks = []
    for _ in range(num_samples):
        img = np.random.rand(IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
        mask = np.random.randint(0, 2, (IMG_SIZE, IMG_SIZE, 1)).astype(np.float32)

        images.append(torch.tensor(img).permute(2, 0, 1))
        masks.append(torch.tensor(mask).permute(2, 0, 1))

    images = torch.stack(images)
    masks = torch.stack(masks)

    # Зберігаємо для наступних запусків
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    torch.save({"images": images, "masks": masks}, DATA_PATH)
    print("💾 Датасет збережено.")

# === Даталоадери ===
dataset = TensorDataset(images, masks)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Модель, втрата, оптимізатор ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Тренування ===
model.train()
for epoch in range(EPOCHS):
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
    total_loss = 0
    for images_batch, masks_batch in loop:
        images_batch = images_batch.to(device)
        masks_batch = masks_batch.to(device)

        outputs = model(images_batch)
        loss = criterion(outputs, masks_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"📊 Епоха {epoch+1}: Середні втрати = {total_loss / len(train_loader):.4f}")

# === Збереження моделі ===
torch.save(model.state_dict(), "unet_model.pth")
print("✅ Модель збережена у файл unet_model.pth")
