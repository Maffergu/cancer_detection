import torch
from torch.cuda.amp import autocast, GradScaler
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim

from PIL import Image
import os
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from multiprocessing import freeze_support

# ======== RCAN DEFINICIÓN ========
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class RCAB(nn.Module):
    def __init__(self, channel):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1),
            CALayer(channel)
        )

    def forward(self, x):
        return self.body(x) + x

class RCAN(nn.Module):
    def __init__(self, num_blocks=10, channel=64, scale=2):
        super(RCAN, self).__init__()
        self.head = nn.Conv2d(1, channel, 3, padding=1)
        self.body = nn.Sequential(*[RCAB(channel) for _ in range(num_blocks)])
        self.upsample = nn.Sequential(
            nn.Conv2d(channel, channel * scale**2, 3, padding=1),
            nn.PixelShuffle(scale)
        )
        self.tail = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = x + res
        x = self.upsample(x)
        x = self.tail(x)
        return x

# ======== DATASET .pgm ========
class PGM_Dataset(Dataset):
    def __init__(self, folder, scale=2):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pgm")]
        self.scale = scale
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        hr = self.to_tensor(img)
        lr_img = T.Resize((hr.shape[1] // self.scale, hr.shape[2] // self.scale), interpolation=Image.BICUBIC)(img)
        lr = self.to_tensor(lr_img)
        return lr, hr

# ======== VISUALIZACIÓN ========
def mostrar_resultado(modelo, dataset, device, idx=0, carpeta_output="output", carpeta_imgs="imgs", acumulador=None):
    modelo.eval()
    os.makedirs(carpeta_output, exist_ok=True)
    os.makedirs(carpeta_imgs, exist_ok=True)

    to_pil = T.ToPILImage()
    lr, hr = dataset[idx]

    path_img = dataset.paths[idx]
    base_name = os.path.splitext(os.path.basename(path_img))[0]
    carpeta_img = os.path.join(carpeta_imgs, base_name)
    os.makedirs(carpeta_img, exist_ok=True)

    with torch.no_grad():
        sr = modelo(lr.unsqueeze(0).to(device)).cpu().squeeze(0)

    sr_np = torch.clamp(sr, 0, 1).squeeze().numpy()
    hr_np = hr.squeeze().numpy()

    psnr = compare_psnr(hr_np, sr_np, data_range=1.0)
    ssim = compare_ssim(hr_np, sr_np, data_range=1.0)

    if acumulador is not None:
        acumulador['total_psnr'] += psnr
        acumulador['total_ssim'] += ssim
        acumulador['n'] += 1

    lr_img = to_pil(lr)
    sr_img = to_pil(torch.clamp(sr, 0, 1))
    hr_img = to_pil(hr)

    lr_img.save(os.path.join(carpeta_img, f"{base_name}_LR.png"))
    sr_img.save(os.path.join(carpeta_img, f"{base_name}_SR.png"))
    hr_img.save(os.path.join(carpeta_img, f"{base_name}_HR.png"))
    print(f"📁 Imágenes guardadas en '{carpeta_img}'")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, image, title in zip(axes, [lr_img, sr_img, hr_img], ['LR', 'SR', 'HR']):
        ax.imshow(image, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    fig.suptitle(f"PSNR: {psnr:.2f} dB   SSIM: {ssim:.4f}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    ruta_guardado = os.path.join(carpeta_output, f"{base_name}_comparacion.png")
    plt.savefig(ruta_guardado)
    print(f"📊 Gráfica guardada como '{ruta_guardado}'")
    #plt.show()

# ======== EJECUCIÓN PRINCIPAL ========
def main():
    # Configuración de dispositivo
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 Dispositivo en uso: {device}, CUDA versión: {torch.version.cuda}")

    # Modelo y datos
    model = RCAN().to(device)
    dataset = PGM_Dataset("inputs")
    num_train = int(len(dataset) * 0.8)
    num_val = len(dataset) - num_train
    train_set, val_set = random_split(dataset, [num_train, num_val])

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=1, num_workers=2, pin_memory=True)

    # Optimizador y scaler
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    # Early stopping
    best_psnr = 0.0
    patience = 10
    counter = 0

    # Entrenamiento
    for epoch in range(100):
        model.train()
        total_loss = 0.0
        for lr, hr in train_loader:
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
            with autocast():
                sr = model(lr)
                loss = F.mse_loss(sr, hr)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

        # Validación
        model.eval()
        total_psnr = 0.0
        with torch.no_grad():
            for lr_val, hr_val in val_loader:
                lr_val, hr_val = lr_val.to(device), hr_val.to(device)
                sr_val = model(lr_val)
                sr_np = torch.clamp(sr_val.squeeze(), 0, 1).cpu().numpy()
                hr_np = hr_val.squeeze().cpu().numpy()
                total_psnr += compare_psnr(hr_np, sr_np, data_range=1.0)
        avg_psnr = total_psnr / len(val_loader)
        print(f"📊 PSNR de validación: {avg_psnr:.2f} dB")

        # Early Stopping
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            counter = 0
            torch.save(model.state_dict(), "mejor_modelo.pth")
            print("✅ Nuevo mejor modelo guardado.")
        else:
            counter += 1
            print(f"⏳ Sin mejora. Paciencia: {counter}/{patience}")
            if counter >= patience:
                print("🛑 Early stopping activado.")
                break

    # Cargar mejor modelo
    model.load_state_dict(torch.load("mejor_modelo.pth"))
    model.eval()

    # Evaluación final y guardado de imágenes
    metricas_acumuladas = {'total_psnr':0.0, 'total_ssim':0.0, 'n':0}
    for i in range(len(dataset)):
        mostrar_resultado(model, dataset, device, idx=i, acumulador=metricas_acumuladas)

    # Métricas promedio
    n = metricas_acumuladas['n']
    avg_psnr = metricas_acumuladas['total_psnr'] / n
    avg_ssim = metricas_acumuladas['total_ssim'] / n
    print(f"\n📈 PSNR promedio: {avg_psnr:.2f} dB")
    print(f"📈 SSIM promedio: {avg_ssim:.4f}")

if __name__ == "__main__":
    freeze_support()
    main()