import os

import matplotlib.pyplot as plt
import plenoptic as po
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

# Load the pre-trained Inception v3 model
inception = models.inception_v3(pretrained=True, transform_input=False)

# Set the model to evaluation mode
inception.eval()
po.tools.remove_grad(inception)


# Define a truncated Inception model
class TruncatedInception(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        # Disable the auxiliary classifiers if they exist
        original_model.AuxLogits = None  # Ensure no auxiliary branches interfere
        # Retain the layers up to 'avgpool'
        self.features = nn.Sequential(
            *list(original_model.children())[
                :-2
            ]  # Exclude the last layer (classifier) and the dropout layer right before
        )
        # Add an explicit adaptive average pooling to maintain correct shape
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # forward pass through the retained layers

        x = self.features(x)

        return x


# Create the truncated model
truncated_inception = TruncatedInception(inception).to("cuda")

############################################################################################################
# Load images into batch


input_folder = "../../Datasets/select_color_textures_unsplash/"
output_folder = "results/"
batch_tensors = []

# Loop through images in  folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
        print(filename)
        # Load images into batch
        with Image.open(input_folder + filename) as img:
            # Bring the image into the correct format
            input_image = Image.open(input_folder + filename)
            preprocess = transforms.Compose(
                [
                    transforms.Resize(299),
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            input_tensor = preprocess(input_image)
            # Add the tensor to the list
            batch_tensors.append(input_tensor)

# Stack all tensors to create a batch
input_batch = torch.stack(batch_tensors)
# print(f"shape of input batch: {input_batch.shape}")

# Move the input batch to GPU if available
if torch.cuda.is_available():
    input_batch = input_batch.to("cuda")


############# for printing images only #############
#  convert images for visualization
def tensor_to_image(tensor):
    tensor = tensor.permute(1, 2, 0).to("cpu")  # Rearrange to (H, W, C) for matplotlib
    return tensor.numpy()


# Create a figure to display images in one row
fig, axes = plt.subplots(1, 12, figsize=(20, 5))  # 1 row, 12 columns

# Display each image
for i, ax in enumerate(axes):
    img = tensor_to_image(input_batch[i])  # Convert each tensor to a NumPy image
    ax.imshow(img)
    ax.axis("off")  # Turn off axis for cleaner display

# Adjust spacing
plt.tight_layout()
plt.savefig(output_folder + "input_imgs_norm_imageNet.png")


# normalize the input again, this time such that it is in the range [0, 1] (required by the Metamer object)
input_batch = (input_batch - input_batch.min()) / (
    input_batch.max() - input_batch.min()
)

# Save image again after normalization to [0, 1]
# Create a figure to display images in one row
fig, axes = plt.subplots(1, 12, figsize=(20, 5))  # 1 row, 12 columns

# Display each image
for i, ax in enumerate(axes):
    img = tensor_to_image(input_batch[i])  # Convert each tensor to a NumPy image
    ax.imshow(img)
    ax.axis("off")  # Turn off axis for cleaner display
# Adjust spacing
plt.tight_layout()
plt.savefig(output_folder + "input_imgs_norm_additional_min_max.png")


############################################################################################################
# Synthesize Metamers


# Parameters
learning_rate = 0.01  # to be passed to the optimizer (default: Adam, lr=0.01),
# but don't know how to get the parameters of metamer object
num_iterations = int(1e5)


met = po.synth.Metamer(input_batch, truncated_inception)
met.synthesize(
    store_progress=False,
    max_iter=num_iterations,
    stop_criterion=1e-18,
    stop_iters_to_check=num_iterations // 1000,
)


# save all metamers and their losses
for i in range(met.metamer.shape[0]):
    po.synth.metamer.plot_synthesis_status(
        batch_idx=i, metamer=met, included_plots=["display_metamer", "plot_loss"]
    )[0].savefig(f"{output_folder}{i}_metamer_loss_n_it_{num_iterations}.png")
