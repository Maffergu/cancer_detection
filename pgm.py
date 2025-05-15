from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T

class PGM_Dataset(Dataset):
    def __init__(self, folder, scale=2):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pgm")]
        self.scale = scale
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')  # Grayscale
        hr = self.to_tensor(img)
        lr = T.Resize((hr.shape[1] // self.scale, hr.shape[2] // self.scale), interpolation=Image.BICUBIC)(img)
        lr = self.to_tensor(lr)
        return lr, hr
