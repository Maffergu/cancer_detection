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
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from itertools import product

# ============================ #
#     CONFIGURACIONES INICIALES #
# ============================ #
BEST_MODEL_PATH = 'best_model_grid.pth'
best_accuracy = 0.0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================ #
#         DATASET              #
# ============================ #
class BreastImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# ============================ #
#    CARGA DEL DATASET         #
# ============================ #
dataset = BreastImageDataset('dataset_augmented.csv', transform=transform)
num_classes = len(dataset.classes)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# ============================ #
#     GRID SEARCH SETUP        #
# ============================ #
param_grid = {
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'batch_size': [16, 32, 48],
    'optimizer_type': ['adam', 'sgd']
}
search_space = list(product(param_grid['learning_rate'], param_grid['batch_size'], param_grid['optimizer_type']))
results = []

# ============================ #
#     FUNCIÓN DE EXPERIMENTO   #
# ============================ #
def run_experiment(lr, batch_size, optimizer_type):
    global best_accuracy

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    model = model.to(device)

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    criterion = nn.CrossEntropyLoss()
    patience = 10
    early_stop_count = 0
    best_local_acc = 0.0

    for epoch in range(80):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluación
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels, all_probs = [], [], []

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

        acc = 100 * correct / total
        if acc > best_local_acc:
            best_local_acc = acc
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= patience:
            break

    # Mostrar resultados del experimento
    print(f"\n🔍 Resultado combinación: LR={lr}, BS={batch_size}, OPT={optimizer_type}")
    print(f"📈 Mejor Accuracy: {best_local_acc:.2f}% en epoch {epoch+1}\n")

    if best_local_acc > best_accuracy:
        best_accuracy = best_local_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"💾 Modelo guardado como mejor hasta ahora con {best_accuracy:.2f}%\n")

        # Guardar métricas clave del mejor modelo
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()

        report = classification_report(all_labels, all_preds, target_names=dataset.classes)
        print("📄 Classification Report:\n", report)

        try:
            binarized_labels = label_binarize(all_labels, classes=list(range(num_classes)))
            auc = roc_auc_score(binarized_labels, np.array(all_probs), average=None)
            for i, score in enumerate(auc):
                print(f"🔵 ROC AUC for class {i}: {score:.4f}")
        except Exception as e:
            print(f"⚠️ ROC AUC could not be computed: {e}")

    return best_local_acc

# ============================ #
#     EJECUTAR GRID SEARCH     #
# ============================ #
for lr, bs, opt in search_space:
    acc = run_experiment(lr, bs, opt)
    results.append((lr, bs, opt, acc))

results.sort(key=lambda x: x[3], reverse=True)
print("\n📊 Resultados del Grid Search:\n")
for i, (lr, bs, opt, acc) in enumerate(results):
    print(f"{i+1:02d}) LR={lr}, BS={bs}, OPT={opt} -> Accuracy: {acc:.2f}%")

pd.DataFrame(results, columns=["Learning Rate", "Batch Size", "Optimizer", "Accuracy"]).to_csv("grid_results.csv", index=False)
print("\n✅ Grid search completo. Resultados y mejores métricas guardadas.")
