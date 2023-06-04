# %%
import torch
from PIL import Image
from typing import List, TextIO
from pathlib import Path
from lavis.models import load_model_and_preprocess
from tqdm import tqdm


# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# define constant path roots
IMAGES_ROOT = Path("../data/images")
OUTPUT_ROOT = Path("output")
OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)

# Define functions to load and process image
def load_raw_image(image_path: str | Path):
    raw_image = Image.open(image_path).convert("RGB")
    return raw_image

def process_image(raw_image, vis_processors):
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    return image



# %%
# Define images to test on
images_path = [
    "00008_Sega-Game-Gear-WB.png",
    "00018_Game-master-console-image.png",
    "00156_The_art_of_video_games_exhibition_crowd.jpg",
    "00450_Nintendo-DS-Lite-Black-Open.jpg"
]

images_path = [Path(image_path) for image_path in images_path]

# Load images
raw_images = [load_raw_image(IMAGES_ROOT / image_path) for image_path in images_path]

# Create one file for each image for the output
files: List[TextIO] = []
for image_path, raw_image in zip(images_path, raw_images):
    # Save raw image
    raw_image.save(OUTPUT_ROOT / image_path)

    # Create file for raw image results
    files.append(open(OUTPUT_ROOT / f"{image_path.stem}.txt", "w"))


# %%
# Define models to test

models_info: List[dict] = [
    {
        "name": "blip2_opt",
        "model_type": "pretrain_opt2.7b"
    },
    {
        "name": "blip2_opt",
        "model_type": "caption_coco_opt2.7b"
    },
    {
        "name": "blip2_opt",
        "model_type": "pretrain_opt6.7b"
    },
    {
        "name": "blip2_opt",
        "model_type": "caption_coco_opt6.7b"
    },
    {
        "name": "blip2_t5",
        "model_type": "pretrain_flant5xl"
    },
    {
        "name": "blip2_t5",
        "model_type": "caption_coco_flant5xl"
    },
    {
        "name": "blip2_t5",
        "model_type": "pretrain_flant5xxl"
    },
]


# %%
# Run tests

progress_bar = tqdm(total=len(models_info) * len(raw_images))

for model_info in models_info:

    # Get model info
    name = model_info["name"]
    model_type = model_info["model_type"]

    # Load model and preprocessors
    model, vis_processor, _ = load_model_and_preprocess(name=name, model_type=model_type, is_eval=True, device=device)

    # Run tests
    for raw_image, file in zip(raw_images, files):

        progress_bar.set_description(f"Model: {name} - {model_type} / Image: {raw_image}")

        # Process image and generate caption
        image = process_image(raw_image, vis_processor)
        caption = model.generate({"image": image, "prompt": "a photo of"})

        # Write results to file
        file.write("========================================\n")
        file.write(f"Model: {name} - {model_type}\n")
        file.write(f"Caption: {caption}\n\n")

        progress_bar.update(1)

        

    
for file in files:
    file.close()
