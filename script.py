from PIL import Image
from diffusers.models.autoencoder_kl import AutoencoderKL
import numpy as np
import torch
import glob
from tqdm import tqdm
import os

images_path = "data" #"Breast_MRI_009_0000_slice95.png"
model_string_list = ["stabilityai/stable-diffusion-2-1-base", "stabilityai/stable-diffusion-xl-base-1.0", "runwayml/stable-diffusion-v1-5"]

# Init VAE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


for model_string in model_string_list:

    # create output folder
    output_folder = f"output/{model_string.replace('/', '_')}"
    os.makedirs(output_folder, exist_ok=True)
    vae = AutoencoderKL.from_pretrained(model_string, subfolder="vae")
    vae = vae.to(device)
    vae.eval()

    # iterate over png files in image_path
    for image_path in tqdm(glob.glob(f"{images_path}/*.png")):

        # preprocess input image
        image = Image.open(image_path)
        # image to torch
        image = torch.from_numpy(np.array(image)).float()
        image = image.unsqueeze(0).unsqueeze(0)
        image = image.repeat_interleave(3, dim=1)
        image = image / 255.0
        image = image * 2.0 - 1.0

        # use VAE to encode and decode the image
        with torch.no_grad():
            latent = vae.encode(image.to(device)).latent_dist.sample()
            recon = vae.decode(latent, return_dict=False)[0]

        # post-process the reconstructed image
        recon = recon.cpu().numpy()
        recon = recon.squeeze(0)[0]
        image = image.cpu().numpy()
        image = image.squeeze(0)[0]
        image = np.concatenate((image, recon), axis=1)
        image = (image + 1.0) / 2.0
        image = image * 255.0
        image = image.astype(np.uint8)
        image = Image.fromarray(image)

        # save image
        output_image_path = f"{output_folder}/{image_path.split('/')[-1].replace('.png', '_recon.png')}"
        image.save(output_image_path)
        print(f"Saved {output_image_path}")