import torch

import pyxu.runtime as pxrt
from diffusers import AutoencoderTiny, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

def encode_prompt(prompt, tokenizer, text_encoder, torch_device):
    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    uncond_input = tokenizer([""], padding="max_length", max_length=text_input.input_ids.shape[-1], return_tensors="pt")
    # Encode the prompt
    with torch.no_grad():
        prompt_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

    return torch.cat([uncond_embeddings, prompt_embeddings])


def to_diffusers(arr):
    r"""Normalize the input array to [-1, 1]"""
    min_ = arr.min()
    max_ = arr.max()
    # First to [0, 1]
    arr -= min_
    arr /= max_
    return (arr - 0.5) * 2.0, min_, max_


def to_pyxu(arr, min_, max_):
    r"""Normalize the input array to [min, max]"""
    return max_ * (arr + 1.0) / 2.0 + min_


def to_m1_1(arr):
    r"""Normalize the input array to [-1, 1]"""
    return (arr - 0.5) * 2.0


def to_0_1(arr):
    r"""Normalize the input array to [0, 1]"""
    return (arr + 1.0) / 2.0

def load_models(prompt, dtype, CUDA):

    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    if dtype == "float32":
        torch.backends.cuda.matmul.allow_tf32 = True

    # Load models
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch_dtype)
    tokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="tokenizer", torch_dtype=torch_dtype
    )
    text_encoder = CLIPTextModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="text_encoder", torch_dtype=torch_dtype
    )
    unet = UNet2DConditionModel.from_pretrained("nota-ai/bk-sdm-tiny-2m", subfolder="unet", torch_dtype=torch_dtype)
    # Change betas
    scheduler = DDPMScheduler(
        beta_start=0.0001,
        beta_end=0.0002,
        # beta_schedule="scaled_linear"
    )

    # Move the encoder and decoder to GPU if CUDA is enabled
    if CUDA:
        torch_device = "cuda"
        vae.to(torch_device)
        text_encoder.to(torch_device)
        unet.to(torch_device)
    else:
        torch_device = "cpu"

    text_embeddings = encode_prompt(prompt, tokenizer, text_encoder, torch_device).to(torch_dtype)

    return scheduler, unet, vae, text_embeddings, torch_device