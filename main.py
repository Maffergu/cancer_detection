import os
import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def process_image(img_path, model_path, output_folder):
    # Cargar modelo ESRGAN
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=4
    )

    upscaler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False  # True si usas GPU con soporte fp16
    )

    # Leer imagen
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: no se pudo cargar la imagen {img_path}")
        return

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    hr_img = img.copy()
    img = cv2.resize(hr_img, (512, 512), interpolation=cv2.INTER_AREA)

    try:
        output, _ = upscaler.enhance(img, outscale=4)

        # Guardar imagen
        os.makedirs(output_folder, exist_ok=True)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_folder, f'upscaled_{filename}.png')
        cv2.imwrite(out_path, output)
        print(f'{out_path} guardada con tamaño {output.shape[1]}x{output.shape[0]}')

        # Redimensionar para comparación
        sr_resized = cv2.resize(output, (hr_img.shape[1], hr_img.shape[0]), interpolation=cv2.INTER_AREA)
        sr_y = cv2.cvtColor(sr_resized, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        hr_y = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]

        psnr = compare_psnr(hr_y, sr_y, data_range=255)
        ssim = compare_ssim(hr_y, sr_y, data_range=255)

        print(f'PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}')
    except Exception as e:
        print(f"Error procesando la imagen: {e}")


if __name__ == '__main__':
    # Define la ruta de la imagen a procesar
    image_path = 'backend/SR/inputs/mdb323.pgm'
    model_path = 'backend/SR/experiments/pretrained_models/ESRGAN/ESRGAN_PSNR_SRx4_DF2K_official-150ff491.pth'
    output_folder = 'results'

    process_image(image_path, model_path, output_folder)
