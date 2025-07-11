import segmentation_models_pytorch as smp

def get_unetModel():
    """
    Returns a U-Net model with a ResNet34 backbone.
    """
    model = smp.Unet(
        encoder_name="resnet34",  # Encoder architecture
        encoder_weights="imagenet",  # Use pre-trained weights
        in_channels=3,  # Input channels (RGB image)
        classes=1,  # Number of output classes (binary segmentation)
        activation=None,  # No activation function at the output layer
    )
    return model