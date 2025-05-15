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
from torchvision.transforms import RandomCrop
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
        self.scale = scale
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
    def __init__(self, folder, scale=2, patch_size=64):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pgm")]
        self.scale = scale
        self.patch_size = patch_size
        self.augment = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip()
        ])
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        # random crop on HR
        hr_patch = RandomCrop((self.patch_size * self.scale, self.patch_size * self.scale))(img)
        # data augmentation
        hr_patch = self.augment(hr_patch)
        hr = self.to_tensor(hr_patch)
        # generate LR patch by downsampling
        lr_img = hr_patch.resize((self.patch_size, self.patch_size), Image.BICUBIC)
        lr = self.to_tensor(lr_img)
        return lr, hr

# ======== VISUALIZACIÓN ========
def mostrar_resultado(modelo, dataset, device, idx=0, carpeta_output="output", carpeta_imgs="imgs", acumulador=None):
    modelo.eval()
    os.makedirs(carpeta_output, exist_ok=True)
    os.makedirs(carpeta_imgs, exist_ok=True)

    to_pil = T.ToPILImage()
    lr, hr = dataset[idx]
    base = os.path.splitext(os.path.basename(dataset.paths[idx]))[0]
    carpeta = os.path.join(carpeta_imgs, base)
    os.makedirs(carpeta, exist_ok=True)

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

    lr_img.save(os.path.join(carpeta, f"{base}_LR.png"))
    sr_img.save(os.path.join(carpeta, f"{base}_SR.png"))
    hr_img.save(os.path.join(carpeta, f"{base}_HR.png"))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img_, title in zip(axes, [lr_img, sr_img, hr_img], ['LR','SR','HR']):
        ax.imshow(img_, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    fig.suptitle(f"PSNR: {psnr:.2f} dB   SSIM: {ssim:.4f}", fontsize=14)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    salida = os.path.join(carpeta_output, f"{base}_comparacion.png")
    plt.savefig(salida)
    plt.close(fig)

# ======== EJECUCIÓN PRINCIPAL ========
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 Dispositivo en uso: {device}, CUDA versión: {torch.version.cuda}")

    input_folder = "inputs"
    patch_size = 64
    scales = [2, 3, 4]

    for scale in scales:
        print(f"\n=== Entrenando y evaluando modelo x{scale} ===")
        # Datos y modelo
        model = RCAN(scale=scale).to(device)
        dataset = PGM_Dataset(input_folder, scale=scale, patch_size=patch_size)
        num_train = int(len(dataset) * 0.8)
        train_set, val_set = random_split(dataset, [num_train, len(dataset)-num_train])

        train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_set,   batch_size=1, num_workers=2, pin_memory=True)

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scaler = GradScaler()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

        best_psnr = 0.0
        patience = 10
        counter = 0

        # Entrenamiento
        for epoch in range(100):
            model.train()
            total_loss = 0.0
            for lr_batch, hr_batch in train_loader:
                lr_batch, hr_batch = lr_batch.to(device), hr_batch.to(device)
                with autocast():
                    sr_batch = model(lr_batch)
                    loss_mse = F.mse_loss(sr_batch, hr_batch)
                    loss_l1  = F.l1_loss(sr_batch, hr_batch)
                    loss = 0.8 * loss_mse + 0.2 * loss_l1
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            print(f"Scale x{scale} | Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

            # Validación
            model.eval()
            total_val_psnr = 0.0
            with torch.no_grad():
                for lr_val, hr_val in val_loader:
                    lr_val, hr_val = lr_val.to(device), hr_val.to(device)
                    sr_val = model(lr_val)
                    sr_np = torch.clamp(sr_val.squeeze(), 0,1).cpu().numpy()
                    hr_np = hr_val.squeeze().cpu().numpy()
                    total_val_psnr += compare_psnr(hr_np, sr_np, data_range=1.0)
            avg_val_psnr = total_val_psnr / len(val_loader)
            scheduler.step(avg_val_psnr)
            print(f"Scale x{scale} | 📊 PSNR validación: {avg_val_psnr:.2f} dB")

            if avg_val_psnr > best_psnr:
                best_psnr = avg_val_psnr
                counter = 0
                fname = f"mejor_modelo_x{scale}.pth"
                torch.save(model.state_dict(), fname)
                print(f"✅ Mejor modelo x{scale} guardado: {fname}")
            else:
                counter += 1
                print(f"Scale x{scale} | ⏳ Sin mejora. Paciencia {counter}/{patience}")
                if counter >= patience:
                    print(f"Scale x{scale} | 🛑 Early stopping activado.")
                    break

        # Cargar y evaluar mejor modelo
        model.load_state_dict(torch.load(f"mejor_modelo_x{scale}.pth"))
        metricas = {'total_psnr':0.0, 'total_ssim':0.0, 'n':0}
        out_folder = f"output_x{scale}"
        img_folder = f"imgs_x{scale}"
        for i in range(len(dataset)):
            mostrar_resultado(model, dataset, device, idx=i,
                              carpeta_output=out_folder,
                              carpeta_imgs=img_folder,
                              acumulador=metricas)
        n = metricas['n']
        print(f"\nScale x{scale} | 📈 PSNR promedio: {metricas['total_psnr']/n:.2f} dB")
        print(f"Scale x{scale} | 📈 SSIM promedio: {metricas['total_ssim']/n:.4f}")

if __name__ == "__main__":
    freeze_support()
    main()