import os
import torch
from torch.cuda.amp import autocast
torch.backends.cudnn.benchmark = True

import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
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

# ======== DATASET .pgm CON RECORTE ========
class PGM_Dataset(Dataset):
    def __init__(self, folder, scale=2):
        self.paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".pgm")
        ])
        self.scale = scale
        self.to_tensor = T.ToTensor()
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        W, H = img.size
        # recortar a múltiplo exacto de scale
        new_W = (W // self.scale) * self.scale
        new_H = (H // self.scale) * self.scale
        if (new_W, new_H) != (W, H):
            img = img.crop((0, 0, new_W, new_H))
        hr = self.to_tensor(img)
        lr_img = T.Resize(
            (new_H // self.scale, new_W // self.scale),
            interpolation=Image.BICUBIC
        )(img)
        lr = self.to_tensor(lr_img)
        return lr, hr

# ======== VISUALIZACIÓN Y EVALUACIÓN ========
def mostrar_resultado(modelo, dataset, device, idx, carpeta_output, carpeta_imgs, acumulador):
    modelo.eval()
    os.makedirs(carpeta_output, exist_ok=True)
    os.makedirs(carpeta_imgs, exist_ok=True)

    to_pil = T.ToPILImage()
    lr, hr = dataset[idx]
    base_name = os.path.splitext(os.path.basename(dataset.paths[idx]))[0]
    out_folder = os.path.join(carpeta_imgs, base_name)
    os.makedirs(out_folder, exist_ok=True)

    with torch.no_grad(), autocast():
        sr = modelo(lr.unsqueeze(0).to(device)).cpu().squeeze(0)

    # convertir a numpy y eliminar canal
    sr_np = np.squeeze(torch.clamp(sr, 0, 1).numpy())
    hr_np = np.squeeze(hr.numpy())

    # emparejar tamaños si difieren
    h_sr, w_sr = sr_np.shape
    h_hr, w_hr = hr_np.shape
    h_min = min(h_sr, h_hr)
    w_min = min(w_sr, w_hr)
    if (h_sr, w_sr) != (h_hr, w_hr):
        sr_np = sr_np[:h_min, :w_min]
        hr_np = hr_np[:h_min, :w_min]

    psnr = compare_psnr(hr_np, sr_np, data_range=1.0)
    ssim = compare_ssim(hr_np, sr_np, data_range=1.0)

    acumulador['total_psnr'] += psnr
    acumulador['total_ssim'] += ssim
    acumulador['n'] += 1

    lr_img = to_pil(lr)
    sr_img = to_pil(sr.clamp(0,1))
    hr_img = to_pil(hr)

    lr_img.save(os.path.join(out_folder, f"{base_name}_LR.png"))
    sr_img.save(os.path.join(out_folder, f"{base_name}_SR.png"))
    hr_img.save(os.path.join(out_folder, f"{base_name}_HR.png"))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(axes, [lr_img, sr_img, hr_img], ['LR','SR','HR']):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle(f"PSNR: {psnr:.2f} dB   SSIM: {ssim:.4f}", fontsize=14)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    fig.savefig(os.path.join(carpeta_output, f"{base_name}_comparacion.png"))
    plt.close(fig)

# ======== PUNTO DE ENTRADA ========
def main():
    freeze_support()

    input_folder   = "test"
    output_folder  = "test_results"
    checkpoint_path = "mejor_modelo.pth"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 Dispositivo en uso: {device}")

    model = RCAN().to(device)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No se encontró checkpoint en '{checkpoint_path}'")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"✅ Checkpoint cargado desde '{checkpoint_path}'")

    dataset = PGM_Dataset(input_folder)
    print(f"📂 Imágenes encontradas para evaluación: {len(dataset)}")

    metricas = {'total_psnr': 0.0, 'total_ssim': 0.0, 'n': 0}
    for i in range(len(dataset)):
        mostrar_resultado(
            modelo=model,
            dataset=dataset,
            device=device,
            idx=i,
            carpeta_output=output_folder,
            carpeta_imgs=output_folder + "_imgs",
            acumulador=metricas
        )

    n = metricas['n']
    print(f"\nResultados finales sobre {n} imágenes:")
    print(f"📈 PSNR promedio: {metricas['total_psnr']/n:.2f} dB")
    print(f"📈 SSIM promedio: {metricas['total_ssim']/n:.4f}")

if __name__ == "__main__":
    main()
