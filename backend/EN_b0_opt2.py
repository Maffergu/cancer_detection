import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ===========================
# CONFIGURACIONES
# ===========================
results_dir = "results_b0_epoch30"
os.makedirs(results_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===========================
# DATASET PERSONALIZADO
# ===========================
class BreastImageDataset(Dataset):
    def __init__(self, csv_file, class_to_idx=None, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        if class_to_idx:
            self.class_to_idx = class_to_idx
        else:
            self.classes = sorted(self.data['target_class'].unique())
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['ruta_imagen']
        image = Image.open(img_path).convert('RGB')
        label_str = self.data.iloc[idx]['target_class']
        if label_str not in self.class_to_idx:
            raise ValueError(f"Clase '{label_str}' no encontrada en class_to_idx.")
        label = self.class_to_idx[label_str]
        if self.transform:
            image = self.transform(image)
        return image, label

# ===========================
# TRANSFORMACIONES
# ===========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_dataset = BreastImageDataset('dataset_augmented_sr.csv', transform=transform)
test_dataset = BreastImageDataset('rutas_imagenes_test_sr.csv', class_to_idx=train_dataset.class_to_idx, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False)

# ===========================
# MODELO EfficientNet B0
# ===========================
num_classes = len(train_dataset.class_to_idx)
model = EfficientNet.from_pretrained('efficientnet-b0')
model._dropout = nn.Dropout(0.4)
model._fc = nn.Linear(model._fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0008)

num_epochs = 30
best_val_acc = 0.0
train_losses, val_accuracies = [], []

# ===========================
# ENTRENAMIENTO Y EVALUACIÓN
# ===========================
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Train Loss: {avg_train_loss:.4f}")

    model.eval()
    correct, total = 0, 0
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())

    acc = correct / total
    val_accuracies.append(acc)
    print(f"Test Accuracy: {acc * 100:.2f}%")

    if acc > best_val_acc:
        best_val_acc = acc
        torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pth'))

# ===========================
# EVALUAR MEJOR MODELO
# ===========================
model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model.pth')))
model.eval()
y_true_total, y_pred_total, y_scores_total = [], [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)

        y_true_total.extend(labels.cpu().numpy())
        y_pred_total.extend(predicted.cpu().numpy())
        y_scores_total.extend(probs.cpu().numpy())

# ===========================
# MÉTRICAS Y VISUALIZACIONES
# ===========================
class_names = list(train_dataset.class_to_idx.keys())
conf_matrix = confusion_matrix(y_true_total, y_pred_total, normalize='true')
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='.2f', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.title("Normalized Confusion Matrix")
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.close()

# Curvas de pérdida y accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.title("Train Loss")
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Val Accuracy')
plt.title("Validation Accuracy")
plt.savefig(os.path.join(results_dir, "loss_accuracy_curves.png"))
plt.close()

# ROC AUC por clase
y_true_bin = np.eye(len(class_names))[y_true_total]
y_scores_arr = np.array(y_scores_total)
plt.figure(figsize=(10, 8))
for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores_arr[:, i])
    auc = roc_auc_score(y_true_bin[:, i], y_scores_arr[:, i])
    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc:.2f})")
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig(os.path.join(results_dir, "roc_curves.png"))
plt.close()

# Reporte de clasificación
report_str = classification_report(y_true_total, y_pred_total, target_names=class_names)
with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
    f.write(report_str)
