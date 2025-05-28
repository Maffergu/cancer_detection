import os
import cv2
import torch
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# Configuración global
IMAGE_PATH = 'backend/SR/inputs/mdb323.pgm'
ESRGAN_MODEL_PATH = 'backend/SR/experiments/pretrained_models/ESRGAN/ESRGAN_PSNR_SRx4_DF2K_official-150ff491.pth'
CLASSIFIER_MODEL_PATH = 'backend/best_model_grid.pth'
OUTPUT_FOLDER = 'results'
CLASSES = ['CIRC', 'NORM', 'MISC', 'ASYM']  # Ajusta según tus clases reales

def superresolucion(img_path, model_path, output_folder):
    # Inicializar modelo Real-ESRGAN
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upscaler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False
    )

    # Cargar imagen
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {img_path}")
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    hr_img = img.copy()
    img = cv2.resize(hr_img, (512, 512), interpolation=cv2.INTER_AREA)

    # Procesar con Real-ESRGAN
    output, _ = upscaler.enhance(img, outscale=4)

    # Guardar imagen
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(output_folder, f'upscaled_{filename}.png')
    cv2.imwrite(out_path, output)

    print(f'{out_path} guardada con tamaño {output.shape[1]}x{output.shape[0]}')

    # Calcular métricas PSNR y SSIM
    sr_resized = cv2.resize(output, (hr_img.shape[1], hr_img.shape[0]), interpolation=cv2.INTER_AREA)
    sr_y = cv2.cvtColor(sr_resized, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    hr_y = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)[:, :, 0]

    psnr = compare_psnr(hr_y, sr_y, data_range=255)
    ssim = compare_ssim(hr_y, sr_y, data_range=255)
    print(f'PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}')

    return out_path

def clasificar_imagen(image_path, model_path, classes):
    # Preparar imagen
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    # Cargar modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNet.from_name('efficientnet-b0')
    model._fc = torch.nn.Linear(model._fc.in_features, len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Inferencia
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
        predicted_class = classes[predicted_idx]
        confidence = probs[0][predicted_idx].item()

    print(f"\n📸 Imagen clasificada como: **{predicted_class}** con {confidence*100:.2f}% de confianza")

if __name__ == '__main__':
    try:
        upscaled_img_path = superresolucion(IMAGE_PATH, ESRGAN_MODEL_PATH, OUTPUT_FOLDER)
        clasificar_imagen(upscaled_img_path, CLASSIFIER_MODEL_PATH, CLASSES)
    except Exception as e:
        print(f"Error en el procesamiento: {e}")
