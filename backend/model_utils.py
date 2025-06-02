# backend/model_utils.py

import os
import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar el modelo una sola vez
eff_model = EfficientNet.from_name('efficientnet-b0')
eff_model = EfficientNet.from_name('efficientnet-b0')
eff_model._dropout = nn.Dropout(0.4)
eff_model._fc = nn.Linear(eff_model._fc.in_features, 7)
eff_model.load_state_dict(torch.load('best_model.pth', map_location=device))
eff_model.to(device)
eff_model.eval()

# Clases
classes = ['ARCH', 'ASYM', 'CALC', 'CIRC', 'MISC', 'NORM', 'SPIC']

# Transformación para clasificación
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def super_resolve_and_classify(img_path: str, model_path: str, output_folder: str = "static/results"):
    # Cargar modelo ESRGAN
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)

    upscaler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False
    )

    # Leer imagen
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    hr_img = img.copy()
    img = cv2.resize(hr_img, (512, 512), interpolation=cv2.INTER_AREA)

    # Superresolución
    output, _ = upscaler.enhance(img, outscale=4)

    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(output_folder, f'upscaled_{filename}.png')
    cv2.imwrite(out_path, output)

    # Clasificación
    image = Image.open(out_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = eff_model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
        predicted_class = classes[predicted_idx]
        confidence = probs[0][predicted_idx].item()

    return {
        "class": predicted_class,
        "confidence": round(confidence, 2),
        "image_path": out_path
    }
