# ============================ #
#         IMPORTACIONES        #
# ============================ #
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    RocCurveDisplay
)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize

# ============================ #
#     CONFIGURACIONES INICIALES #
# ============================ #
BEST_MODEL_PATH = 'best_model_clas.pth'
best_accuracy = 0.0
num_epochs = 50
patience = 5
epochs_without_improvement = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Comprobar cual es el device
if device.type == 'cuda':
    print(f"🚀 Usando GPU: {torch.cuda.get_device_name(0)}")
else:
    print("🚀 Usando CPU")

# ============================ #
#         DATASET              #
# ============================ #
class BreastImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        # Crear mapeo clase -> índice
        self.classes = self.data['target_class'].unique().tolist()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['ruta_imagen']
        image = Image.open(img_path).convert('RGB')
        
        label_str = self.data.iloc[idx]['target_class']
        label = self.class_to_idx[label_str]
        
        if self.transform:
            image = self.transform(image)
        return image, label

# ============================ #
#       TRANSFORMACIONES       #
# ============================ #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# ============================ #
#    CARGA Y DIVISIÓN DEL DATASET #
# ============================ #
dataset = BreastImageDataset('dataset_augmented.csv', transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ============================ #
#           MODELO             #
# ============================ #
num_classes = len(dataset.classes)
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, num_classes)
model = model.to(device)

# ============================ #
#      OPTIMIZADOR Y LOSS      #
# ============================ #
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============================ #
#        FUNCIONES UTILES      #
# ============================ #
def train():
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
    print(f'Train Loss: {running_loss / len(train_loader):.4f}')

def evaluate(return_preds=False):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\n📊 Test Accuracy: {accuracy:.2f}%")

    return (accuracy, all_labels, all_preds, all_probs) if return_preds else accuracy

# ============================ #
#         LOOP PRINCIPAL       #
# ============================ #
for epoch in range(num_epochs):
    print(f"\n🔁 Epoch {epoch + 1}/{num_epochs}")
    train()
    acc = evaluate()

    if acc > best_accuracy:
        best_accuracy = acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f'💾 Nuevo mejor modelo guardado con {best_accuracy:.2f}% accuracy')
        epochs_without_improvement = 0  # Reiniciar contador
    else:
        epochs_without_improvement += 1
        print(f"⚠️ No hubo mejora. Épocas sin mejora: {epochs_without_improvement}/{patience}")

    # Verificar condición de early stopping
    if epochs_without_improvement >= patience:
        print(f"\n⏹️ Early stopping: no se ha mejorado en {patience} épocas consecutivas.")
        break

# === CARGAR MEJOR MODELO Y GENERAR MÉTRICAS FINALES ===
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()
accuracy, all_labels, all_preds, all_probs = evaluate(return_preds=True)

# === Matriz de confusión ===
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")  # Guardar imagen
plt.close()

# === Reporte de clasificación ===
report = classification_report(all_labels, all_preds, target_names=dataset.classes)
print("📄 Final Classification Report:\n")
print(report)

# === ROC-AUC Multiclase ===
try:
    binarized_labels = label_binarize(all_labels, classes=list(range(num_classes)))
    auc = roc_auc_score(binarized_labels, all_probs, average=None)
    for i, class_name in enumerate(dataset.classes):
        print(f"🔵 ROC AUC for class '{class_name}': {auc[i]:.4f}")
except Exception as e:
    print(f"⚠️ ROC AUC could not be computed: {e}")


