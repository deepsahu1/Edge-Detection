import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class BIPED(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        
        #Loads BIPED images for supervised training.
    
        self.image_dir = image_dir
        self.transform = transform
        self.mask_transform= mask_transform
        self.mask_dir  = mask_dir

        # List all image files (ignores masks)
        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        print(f"✅ Found {len(self.images)} images in {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        

        # Load RGB image
        image = Image.open(image_path).convert("RGB")

        base_name = os.path.splitext(self.images[idx])[0]
        mask_name = f"{base_name}.png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        #Load the mask
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)

        return image, mask
