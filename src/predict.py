import torch
from torchvision import transforms
from PIL import Image, ImageFilter
from src.model import get_unetModel
import os
import torch.nn.functional

def predict(input_dir, model_path, output_dir = "output/edges", results_dir = "results", threshold = .5,  device='cuda' if torch.cuda.is_available() else 'cpu'):
    #Loads a pre-trained U-Net model and predicts edges  on a single image.

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok = True)

    #Load the model
    model = get_unetModel().to(device)
    state = torch.load(model_path, map_location = device, weights_only = True)
    model.load_state_dict(state)
    model.eval()

    #pre-processing the input image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    for filename in sorted(os.listdir(input_dir)):

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        #Run the inference

        with torch.no_grad():
            output = model(input_tensor)
            output = torch.sigmoid(output)
            output = torch.nn.functional.interpolate(
                output,
                size=image.size[::-1],  # (height, width)
                mode="bilinear",
                align_corners = False,
                antialias = True
            )

            edge_map = (output[0].cpu() > threshold).float()

        
       #Save edge map in grayscale
        edge_map_img = transforms.ToPILImage()(edge_map)
        edge_map_img = edge_map_img.filter(ImageFilter.GaussianBlur(radius=1))
        edge_filename = f"{os.path.splitext(filename)[0]}_edge.png"
        edge_save_path = os.path.join(output_dir, edge_filename)
        edge_map_img.save(edge_save_path)

        #Save side by side comparison
        comparison = Image.new("RGB", (image.width*2,image.height))
        comparison.paste(image, (0,0))
        comparison.paste(edge_map_img.convert("RGB"), (image.width,0))
        comp_filename = f"{os.path.splitext(filename)[0]}_comparison.png"
        comp_path = os.path.join(results_dir, comp_filename)
        comparison.save(comp_path)

        print(f"Saved edges: {edge_save_path}")
        print(f"Saved commparion images: {comp_path}")

    print("All edge maps predicted successfully", output_dir)
    print("All comparison images saved successfully", results_dir)


if __name__ == "__main__":

    test_dir = "data/BIPED/imgs/test/rgbr"
    model_weights = "models/unet_biped_epoch_92.pth"

    predict(test_dir, model_weights)



