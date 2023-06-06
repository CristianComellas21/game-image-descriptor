
#%%
from typing import List
from omegaconf import OmegaConf
from lavis.models import load_model_and_preprocess, load_model, load_preprocess
from tqdm import tqdm
from lavis import registry

#%%
models_info: List[dict] = [
    {
        "name": "blip2_opt",
        "model_type": "caption_coco_opt2.7b",
        "checkpoint_path": "../output/caption_game/20230603181_coco_game_finetuned/checkpoint_best.pth",
        "save_name": "blip2_opt_coco_game_finetuned"
    },
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


for model_info in tqdm(models_info):

    # Get model info
    name = model_info["name"]
    model_type = model_info["model_type"]

    checkpoint_path = model_info.get("checkpoint_path", None)

    if checkpoint_path is None:
        # Load model and preprocessors
        model, vis_processor, _ = load_model_and_preprocess(name=name, model_type=model_type, is_eval=True)
    else:
        # Load model and preprocessors from checkpoint
        cfg = OmegaConf.load(registry.get_model_class(name).default_config_path(model_type))
        vis_processor, _ = load_preprocess(cfg.preprocess)
        model = load_model(name=name, model_type=model_type, is_eval=True, checkpoint=checkpoint_path)
        print(f"Loaded checkpoint from {model_info['checkpoint_path']}") 


    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Model: {name} - {model_type}")
    print(f"Trainable params: {trainable_params}")
    print(f"Total params: {total_params}")
# %%
