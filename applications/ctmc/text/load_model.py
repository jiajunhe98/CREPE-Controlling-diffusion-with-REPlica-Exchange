import os

import torch

# Import local utils for load_hydra_config_from_run
import utils
from model import SEDD
from model.ema import ExponentialMovingAverage
from model.transformer import DeftSEDD, FinetuneSEDD


def load_model_hf(dir, device):
    score_model = SEDD.from_pretrained(dir).to(device)
    return score_model


def load_model_local(root_dir, device):
    cfg = utils.load_hydra_config_from_run(root_dir)
    score_model = SEDD(cfg).to(device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)

    ckpt_dir = os.path.join(root_dir, "checkpoints-meta", "checkpoint.pth")
    loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=False)

    score_model.load_state_dict(loaded_state["model"])
    ema.load_state_dict(loaded_state["ema"])

    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    return score_model


def load_model(root_dir, device):
    try:
        return load_model_hf(root_dir, device)
    except:
        return load_model_local(root_dir, device)


def load_finetune_model(root_dir, checkpoint_no, device):
    # .hydra config is in the parent directory of checkpoints
    # Normalize path by removing trailing slashes before checking
    normalized_root = root_dir.rstrip("/")
    config_dir = (
        os.path.dirname(normalized_root)
        if normalized_root.endswith("checkpoints")
        else normalized_root
    )
    cfg = utils.load_hydra_config_from_run(config_dir)
    model = SEDD.from_pretrained("louaaron/sedd-medium")
    model = FinetuneSEDD(model).to(device)
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.training.ema)

    ckpt_dir = os.path.join(root_dir, f"checkpoint_{checkpoint_no}.pth")
    loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=False)

    keys = list(loaded_state["model"].keys())
    for k in keys:
        path = k.split(".")
        new_path = ".".join([path[0], *path[2:]])
        loaded_state["model"][new_path] = loaded_state["model"][k]
        del loaded_state["model"][k]

    model.load_state_dict(loaded_state["model"])
    ema.load_state_dict(loaded_state["ema"])

    ema.store(model.parameters())
    ema.copy_to(model.parameters())

    return model
