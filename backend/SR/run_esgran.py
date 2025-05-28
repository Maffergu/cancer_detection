import os
import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from tqdm import tqdm

# Rutas
input_folder = 'backend/SR/inputs'
output_folder = 'backend/SR/results'
model_path = 'backend/SR/experiments/pretrained_models/ESRGAN/ESRGAN_PSNR_SRx4_DF2K_official-150ff491.pth'

# Crear carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Modelo ESRGAN (RRDBNet)
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

# Inicializar RealESRGANer
upscaler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False  # True si usas GPU con soporte fp16
)

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

total_psnr = 7034.25
total_ssim = 159.8852
count = 162

for img_name in tqdm(os.listdir(input_folder)):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.pgm')):
        continue

    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    if img is None:
        print(f"Warning: no se pudo cargar {img_name}")
        continue

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # HR = imagen original 1024x1024
    hr_img = img.copy()

    # LR = imagen de 512x512
    img = cv2.resize(hr_img, (512, 512), interpolation=cv2.INTER_AREA)

    try:
        # SR = imagen de 2048x2048
        output, _ = upscaler.enhance(img, outscale=4)

        # Guardar imagen
        filename, ext = os.path.splitext(img_name)
        out_path = os.path.join(output_folder, f'upscaled_{filename}.png')
        cv2.imwrite(out_path, output)
        print(f'{out_path} guardada con tamaño {output.shape[1]}x{output.shape[0]}')  # ancho x alto

        # Redimensionar SR a 1024x1024 para comparar con HR
        sr_resized = cv2.resize(output, (hr_img.shape[1], hr_img.shape[0]), interpolation=cv2.INTER_AREA)

        # Convertir a escala de grises (canal Y)
        sr_y = cv2.cvtColor(sr_resized, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        hr_y = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]

        # Calcular métricas
        psnr = compare_psnr(hr_y, sr_y, data_range=255)
        ssim = compare_ssim(hr_y, sr_y, data_range=255)

        print(f'{img_name} - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}')

        total_psnr += psnr
        total_ssim += ssim
        count += 1
        print(f'PSNR acumulado: {total_psnr:.2f} dB, SSIM acumulado: {total_ssim:.4f}, count: {count}')

    except Exception as e:
        print(f"Error procesando {img_name}: {e}")

# Promedio final
if count > 0:
    print(f'\nPromedio PSNR: {total_psnr / count:.2f} dB')
    print(f'Promedio SSIM: {total_ssim / count:.4f}')
else:
    print('No se procesaron imágenes válidas.')

