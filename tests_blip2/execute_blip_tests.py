
# %%
import torch
from PIL import Image
from typing import List, TextIO
from pathlib import Path
from lavis.models import load_model_and_preprocess, load_model, load_preprocess
from lavis import registry
from tqdm import tqdm
from omegaconf import OmegaConf
import json
import numpy as np


# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# define constant path roots
IMAGES_ROOT = Path("../data/images")

OUTPUT_ROOT = Path("output")
OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)

OUTPUT_IMAGES_ROOT = OUTPUT_ROOT / "images"
OUTPUT_IMAGES_ROOT.mkdir(exist_ok=True, parents=True)

OUTPUT_CAPTIONS_ROOT = OUTPUT_ROOT / "captions"
OUTPUT_CAPTIONS_ROOT.mkdir(exist_ok=True, parents=True)


# Define functions to load and process image
def load_raw_image(image_path: str | Path):
    raw_image = Image.open(image_path).convert("RGB")
    return raw_image

def process_image(raw_image, vis_processors):
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    return image



# %%
# Define images to test on

partition = "test"

# load annotations
samples = json.load(open(f"../data/annotations/{partition}.json"))
# selected_samples = np.random.default_rng(seed=21).choice(samples, size=10, replace=False)
selected_samples = samples

# Save caption references
with open(OUTPUT_CAPTIONS_ROOT / "references.json", "w") as f:
    transformed_samples = {}
    for sample in selected_samples:
        key = Path(sample["image"]).stem
        value = [sample["caption"]]
        transformed_samples[key] = value
    json.dump(transformed_samples, f, indent=4)




# %%
# Take only the image paths
images_path = [Path(Path(sample["image"]).name) for sample in selected_samples]


# Load images
raw_images = [load_raw_image(IMAGES_ROOT / image_path) for image_path in images_path]

# Save images
for file in OUTPUT_IMAGES_ROOT.glob("*"):
    file.unlink()

for image, image_path in zip(raw_images, images_path):
    image.save(OUTPUT_IMAGES_ROOT / image_path)




# %%
# Define models to test

models_info: List[dict] = [
    # Fine-tuned models
    {
        "name": "blip2_opt",
        "model_type": "pretrain_opt6.7b",
        "checkpoint_path": "../output/caption_game/20230604185_opt_6_game_finetuned/checkpoint_best.pth",
        "save_name": "blip2_opt_6_game_finetuned"
    },
    # {
    #     "name": "blip2_opt",
    #     "model_type": "pretrain_opt2.7b",
    #     "checkpoint_path": "../output/caption_game/20230604200_opt_2_game_finetuned/checkpoint_best.pth",
    #     "save_name": "blip2_opt_2_game_finetuned"
    # },
    # {
    #     "name": "blip2_opt",
    #     "model_type": "caption_coco_opt2.7b",
    #     "checkpoint_path": "../output/caption_game/20230611122_coco_opt_2_game_finetuned_loadft/checkpoint_best.pth",
    #     "save_name": "blip2_coco_opt_2_game_finetuned_loadft"
    # },
    # {
    #     "name": "blip2_opt",
    #     "model_type": "caption_coco_opt6.7b",
    #     "checkpoint_path": "../output/caption_game/20230611125_coco_opt_6_game_finetuned_loadft/checkpoint_best.pth",
    #     "save_name": "blip2_coco_opt_6_game_finetuned_loadft"
    # },
    # {
    #     "name": "blip2_t5",
    #     "model_type": "pretrain_flant5xl",
    #     "checkpoint_path": "../output/caption_game/20230605195_t5_xl_game_finetuned/checkpoint_best.pth",
    #     "save_name": "blip2_t5_xl_game_finetuned"
    # },
    # {
    #     "name": "blip2_t5",
    #     "model_type": "pretrain_flant5xxl",
    #     "checkpoint_path": "../output/caption_game/20230611120_t5_xxl_game_finetuned_freezed/checkpoint_best.pth",
    #     "save_name": "blip2_t5_xxl_game_finetuned_freezed"
    # },
    # {
    #     "name": "blip2_t5",
    #     "model_type": "caption_coco_flant5xl",
    #     "checkpoint_path": "../output/caption_game/20230611133_coco_t5_xl_game_finetuned_loadft/checkpoint_best.pth",
    #     "save_name": "blip2_coco_t5_xl_game_finetuned_loadft"
    # },
    # # Not fine-tuned models -------------------------------------------------
    # {
    #     "name": "blip2_opt",
    #     "model_type": "pretrain_opt2.7b"
    # },
    # {
    #     "name": "blip2_opt",
    #     "model_type": "caption_coco_opt2.7b"
    # },
    # {
    #     "name": "blip2_opt",
    #     "model_type": "pretrain_opt6.7b"
    # },
    # {
    #     "name": "blip2_opt",
    #     "model_type": "caption_coco_opt6.7b"
    # },
    # {
    #     "name": "blip2_t5",
    #     "model_type": "pretrain_flant5xl"
    # },
    # {
    #     "name": "blip2_t5",
    #     "model_type": "caption_coco_flant5xl"
    # },
    # {
    #     "name": "blip2_t5",
    #     "model_type": "pretrain_flant5xxl"
    # },
]


# %%
# Run tests

progress_bar = tqdm(total=len(models_info) * len(raw_images))

for model_info in models_info:

    # Get model info
    name = model_info["name"]
    model_type = model_info["model_type"]

    checkpoint_path = model_info.get("checkpoint_path", None)

    if checkpoint_path is None:
        # Load model and preprocessors
        model, vis_processor, _ = load_model_and_preprocess(name=name, model_type=model_type, is_eval=True, device=device)
    else:
        # Load model and preprocessors from checkpoint
        cfg = OmegaConf.load(registry.get_model_class(name).default_config_path(model_type))
        vis_processor, _ = load_preprocess(cfg.preprocess)
        model = load_model(name=name, model_type=model_type, is_eval=True, device=device, checkpoint=checkpoint_path)
        print(f"Loaded checkpoint from {model_info['checkpoint_path']}") 

    # Run tests
    captions = {}
    for raw_image, path in zip(raw_images, images_path):

        progress_bar.set_description(f"Model: {name} - {model_type} / Image: {path.stem}")

        # Process image and generate caption
        image = process_image(raw_image, vis_processor)
        caption = model.generate({"image": image, "prompt": "a photo of"})[0]

        # Save caption in object
        captions[path.stem] = caption


        progress_bar.update(1)

    # Save captions
    save_name = model_info.get("save_name", f"{name}_{model_type}")
    with open(OUTPUT_CAPTIONS_ROOT / f"{save_name}.json", "w") as f:
        json.dump(captions, f, indent=4)

progress_bar.close()