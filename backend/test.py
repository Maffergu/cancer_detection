import os
import torch
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

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

# ======== Dataset de solo inferencia (sin degradación) ========
class PGMInferenceDataset(Dataset):
    def __init__(self, folder, scale=2):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pgm")]
        self.scale = scale
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        W, H = img.size
        new_W = (W // self.scale) * self.scale
        new_H = (H // self.scale) * self.scale
        if (W, H) != (new_W, new_H):
            img = img.crop((0, 0, new_W, new_H))

        lr = self.to_tensor(img)
        name = os.path.splitext(os.path.basename(self.paths[idx]))[0]
        return lr, name

# ======== Inferencia ========
def main():
    input_folder = "inputs"
    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Inference device: {device}")

    scale = 2
    model = RCAN(scale=scale).to(device)
    model.load_state_dict(torch.load("mejor_modelo_x2.pth", map_location=device))
    model.eval()

    dataset = PGMInferenceDataset(input_folder, scale=scale)
    to_pil = T.ToPILImage()

    with torch.no_grad():
        for lr, name in tqdm(dataset, desc="Procesando imágenes"):
            lr = lr.unsqueeze(0).to(device)
            sr = model(lr).clamp(0, 1).cpu().squeeze(0)
            sr_img = to_pil(sr)
            sr_img.save(os.path.join(output_folder, f"{name}_SR.png"))

    print(f"✅ Imágenes guardadas en la carpeta '{output_folder}'")

if __name__ == "__main__":
    main()
