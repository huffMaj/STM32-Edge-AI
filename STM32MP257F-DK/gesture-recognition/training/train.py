import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# CONFIG
DATASET_PATH = "dataset"
MODEL_SAVE_PATH = "models/gesture_model.pth"
ACCURACY_PLOT_PATH = "models/train_val_accuracy.png"
CONFUSION_MATRIX_PLOT_PATH = "models/confusion_matrix.png"
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-4
NUM_CLASSES = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

# DATASET
dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
class_names = dataset.classes

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# MODEL
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

train_acc_history = []
val_acc_history = []

# TRAIN LOOP
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_correct = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()

    train_acc = train_correct / train_size

    # VALIDATION
    model.eval()
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()

    val_acc = val_correct / val_size

    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Model saved.")

# ACCURACY CURVE
epochs = range(1, EPOCHS + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc_history, marker="o", label="Train Accuracy")
plt.plot(epochs, val_acc_history, marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Validation Accuracy")
plt.ylim(0, 1)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(ACCURACY_PLOT_PATH, dpi=150)
plt.close()
print(f"Accuracy plot saved: {ACCURACY_PLOT_PATH}")

# CONFUSION MATRIX ON VALIDATION SET
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.tolist())

cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(7, 7))
disp.plot(cmap="Blues", ax=ax, colorbar=False)
plt.title("Confusion Matrix (Validation)")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_PLOT_PATH, dpi=150)
plt.close(fig)
print(f"Confusion matrix saved: {CONFUSION_MATRIX_PLOT_PATH}")
