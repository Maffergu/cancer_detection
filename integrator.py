import os
import torch
from PIL import Image
import torchvision.transforms as T
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.nn.functional as F

# ======== RCAN DEFINICIÓN ========
class CALayer(torch.nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(channel, channel // reduction, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(channel // reduction, channel, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class RCAB(torch.nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(channel, channel, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(channel, channel, 3, padding=1),
            CALayer(channel),
            torch.nn.Dropout2d(0.1)
        )

    def forward(self, x):
        return self.body(x) + x

class RCAN(torch.nn.Module):
    def __init__(self, num_blocks=16, channel=64, scale=2):
        super().__init__()
        self.head = torch.nn.Conv2d(1, channel, 3, padding=1)
        self.body = torch.nn.Sequential(*[RCAB(channel) for _ in range(num_blocks)])
        self.upsample = torch.nn.Sequential(
            torch.nn.Conv2d(channel, channel * scale**2, 3, padding=1),
            torch.nn.PixelShuffle(scale)
        )
        self.tail = torch.nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        x = self.upsample(x)
        x = self.tail(x)
        return x

# ======== CLASIFICADOR ========
class Classifier:
    def __init__(self, model_path, num_classes, device):
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.device = device

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def predict(self, img_rgb):
        img = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(img)
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
        return pred_class, probs.squeeze().cpu().numpy()

# ======== PROCESO COMPLETO ========
def main():
    # === Configuración ===
    input_image_path = "inputs/mdb001.pgm"  # <- Cambia a tu ruta de imagen
    rcan_model_path = "mejor_modelo_x2.pth"
    classifier_model_path = "best_model_grid.pth"
    output_image_path = "salida001_SR.png"
    class_names = ['CIRC', 'NORM', 'MISC', 'ASYM']  # <- Ajusta según tu dataset
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Usando dispositivo: {device}")

    # === RCAN ===
    rcan = RCAN(scale=2).to(device)
    rcan.load_state_dict(torch.load(rcan_model_path, map_location=device))
    rcan.eval()

    # === Clasificador ===
    classifier = Classifier(classifier_model_path, num_classes, device)

    # === Leer y preprocesar imagen ===
    img = Image.open(input_image_path).convert("L")
    W, H = img.size
    new_W = (W // 2) * 2
    new_H = (H // 2) * 2
    img = img.crop((0, 0, new_W, new_H))

    to_tensor = T.ToTensor()
    lr = to_tensor(img).unsqueeze(0).to(device)

    # === Superresolución ===
    with torch.no_grad():
        sr = rcan(lr).clamp(0, 1).cpu().squeeze(0)

    sr_img = T.ToPILImage()(sr)
    sr_img.save(output_image_path)
    print(f"✅ Imagen SR guardada en: {output_image_path}")

    # === Convertir a RGB para EfficientNet ===
    sr_img_rgb = sr_img.convert("RGB")

    # === Clasificación ===
    pred_class, probs = classifier.predict(sr_img_rgb)
    print(f"🔍 Clase predicha: {class_names[pred_class]} ({probs[pred_class]*100:.2f}%)")

if __name__ == "__main__":
    main()
