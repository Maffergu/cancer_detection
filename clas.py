# ============================ #
#     CLASIFICACIÓN DE IMAGEN #
# ============================ #
import torch
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image

# Ruta a la imagen superresuelta generada por Real-ESRGAN
superres_image_path = 'results/upscaled_mdb323.png'  # ajusta si usas otro nombre

# Transformaciones para EfficientNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Cargar clase y etiquetas
classes = ['CIRC', 'NORM', 'MISC', 'ASYM']  # reemplaza con las verdaderas clases
num_classes = len(classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar modelo EfficientNet
model = EfficientNet.from_name('efficientnet-b0')
model._fc = torch.nn.Linear(model._fc.in_features, num_classes)
model.load_state_dict(torch.load('backend/best_model_grid.pth', map_location=device))
model.to(device)
model.eval()

# Cargar imagen y preprocesar
image = Image.open(superres_image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)

# Clasificar
with torch.no_grad():
    outputs = model(image_tensor)
    probs = torch.softmax(outputs, dim=1)
    predicted_idx = torch.argmax(probs, dim=1).item()
    predicted_class = classes[predicted_idx]
    confidence = probs[0][predicted_idx].item()

print(f"\n📸 Imagen clasificada como: **{predicted_class}** con {confidence*100:.2f}% de confianza")
