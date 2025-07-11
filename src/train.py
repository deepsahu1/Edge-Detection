import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import BIPED
from src.model import get_unetModel
import os

def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat*target_flat).sum()
    return 1 - ((2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

def train(
    image_dir,
    mask_dir,
    num_epochs=100,
    batch_size=4,
    learning_rate=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    #Fine-tuning the model on the BSDS500 dataset

    #Transforms for input images and masks
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 1.5)),  #  Add Gaussian Blur here
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5])
])

    #Transforms for masks (no normalization)
    mask_transform  = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    os.makedirs("models", exist_ok=True)

    #Dataset and DataLoader
    dataset = BIPED(image_dir, mask_dir, transform=transform, mask_transform = mask_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #Model, Loss function, Optimizer
    model = get_unetModel().to(device)
    pos_weight = torch.tensor([10.0]).to(device)  # Boost edge pixel importance
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Training on {device} for {num_epochs} epochs...")

    #Loop for training
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
             
            outputs = model(images)
            
            loss_bce = criterion(outputs, masks)
            loss_dice = dice_loss(outputs,masks)
            loss = loss_bce + loss_dice
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        print(f"BCE Loss: {loss_bce.item():.4f}, Dice Loss: {loss_dice.item():.4f}")

        #Save the model
        
        model_path = f"models/unet_biped_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    image_dir = "data/BIPED/imgs/train/rgbr/real/"
    mask_dir = "data/BIPED/edge_maps/train/rgbr/real/"
    train(image_dir, mask_dir)