
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


# Define functions to load and process image
def load_raw_image(image_path: str | Path):
    raw_image = Image.open(image_path).convert("RGB")
    return raw_image

def process_image(raw_image, vis_processors):
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    return image

def black_background_thumbnail(path_to_image, thumbnail_size=(224,224)):
    background = Image.new('RGB', thumbnail_size, "black")    
    source_image = Image.open(path_to_image).convert("RGB")
    source_image.thumbnail(thumbnail_size)
    (w, h) = source_image.size
    background.paste(source_image, ((thumbnail_size[0] - w) // 2, (thumbnail_size[1] - h) // 2 ))
    return background





# %%
# Define models to test

# model_info = {
#         "name": "blip2_opt",
#         "model_type": "pretrain_opt6.7b",
#         "checkpoint_path": "../output/caption_game/20230604185_opt_6_game_finetuned/checkpoint_best.pth",
#         "save_name": "blip2_opt_6_game_finetuned"
#     }

model_info = {
        "name": "blip2_opt",
        "model_type": "pretrain_opt6.7b"
    }

# models_info: List[dict] = [
#     # Fine-tuned models
#     {
#         "name": "blip2_opt",
#         "model_type": "pretrain_opt6.7b",
#         "checkpoint_path": "../output/caption_game/20230604185_opt_6_game_finetuned/checkpoint_best.pth",
#         "save_name": "blip2_opt_6_game_finetuned"
#     },
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
# ]


# %%
# Run tests
## Load model
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



# %%
# Take only the image paths
image_path = "robotic_animals.png"


# Load image
raw_image = black_background_thumbnail(image_path)



# Process image and generate caption
image = process_image(raw_image, vis_processor)
caption = model.generate({"image": image, "prompt": "a photo of"})[0]

# Print caption
print(f"Caption for image {image_path}: {caption}")
# %%
