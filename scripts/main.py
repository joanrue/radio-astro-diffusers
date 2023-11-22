import matplotlib.pyplot as plt
import scipy.signal as ss

from PIL import Image
import torch
import numpy as np
import astropy.units as u

import pyxu.opt.stop as pxst
import pyxu.runtime as pxrt
from pyxu.abc.operator import DiffFunc, ProxFunc
from pyxu.operator import SquaredL2Norm
from pyxu.operator.interop import from_torch
from pyxu.opt.solver import PGD

from radio_utils import create_forward, create_dataset, add_noise, show_image
from diffusers_utils import load_models, to_0_1, to_m1_1

# Set constants and data types
CUDA = True
lambda_ = 0  # 1e-4  # Ridge
dtype = "float32"
acceleration = False
SNR = 15  # dB
VIDEO = True
max_iter = 1  # iterations per each scheduler variance value
stop_crit = pxst.MaxIter(max_iter)

prompt = ["A galaxy that looks like a face"]
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
num_inference_steps = 100  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance
generator = torch.manual_seed(0)  # Seed generator to create the initial latent noise

# Adjust some parameters accordingly
if VIDEO:
    import os

    import imageio

    frame_step = 20
if CUDA:
    import cupy as xp
else:
    import numpy as xp


# Load the input image and normalize it

# Set random seed and define the dimensions of the input data
generator = torch.manual_seed(0)
batch_size = 1
height, width = (512, 512)
in_height, in_width = height // 8, width // 8
in_channels = 4

precision = pxrt.Width.SINGLE if dtype == "float32" else pxrt.Width.DOUBLE
torch_dtype = torch.float32 if dtype == "float32" else torch.float64

scheduler, unet, vae, text_embeddings, torch_device = load_models(prompt, dtype, CUDA)
# Measure the data
sky_image, visibilities = create_dataset(
    name="LOWBD2",
    rmax=3_000.0,
    ra=+15.0 * u.deg,
    dec=-45.0 * u.deg)

x = sky_image.pixels.data.squeeze()

# Upsample image to be 512, 512
from scipy.ndimage import zoom
upsample = 2
x = xp.asarray(np.tile(zoom(x, (upsample, upsample)), (3, 1, 1))).clip(0, 1)

# Set the precision for the reconstruction
with pxrt.Precision(precision):
    forward = create_forward(sky_image, visibilities, upsample=upsample, repeats=3)
    y = forward(x.ravel())
    y, sigma = add_noise(y, SNR)

    # Create a directory to store the images
    image_folder = f"../results/"
    os.makedirs(image_folder, exist_ok=True)
    # Remove existing images from previous experiment
    [os.remove(os.path.join(image_folder, filename)) for filename in os.listdir(image_folder)]

    # # Set the initial guess for the latent space
    # latents = torch.randn((1, in_channels, in_height, in_width), generator=generator)
    # latents = latents.to(torch_device).to(torch_dtype)
    # latents = latents * scheduler.init_noise_sigma
    #
    # # Map it to the image space
    # with torch.no_grad():
    #     x0 = vae.decode(latents).sample
    # from pyxu.operator.interop.torch import _from_torch
    #
    # x0 = _from_torch(x0)
    # x0 = to_0_1(x0)

    x0 = forward.adjoint(y.ravel())

    # Loss function
    sl2norm = SquaredL2Norm(dim=y.size).asloss(y.ravel())
    # loss = 1 / (2 * sigma**2) * sl2norm * forward
    loss = 1e3 * sl2norm * forward
    loss.diff_lipschitz = sl2norm.estimate_diff_lipschitz()
    print(f"Estimated Lipschitz constant: {loss.diff_lipschitz}")

    # Regularization
    ridge_reg = lambda_ * SquaredL2Norm(dim=x0.size)

    scheduler.set_timesteps(num_inference_steps)
    counter = 0
    for i, t in enumerate(scheduler.timesteps):

        def flat_denoiser(tensor, tau):
            sh = tensor.shape[:-3]
            tensor = to_m1_1(tensor)
            # From grayscale to RBG
            tensor = tensor.reshape(1, 3, height, width)
            latents = vae.encode(tensor).latents

            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            step = scheduler.step(noise_pred, t, latents)
            xo_hat = step.pred_original_sample
            xo_hat = vae.decode(xo_hat).sample
            xo_hat = to_0_1(xo_hat)
            return xo_hat.reshape(*sh, 3 * height * width)

        op_denoiser = from_torch(
            apply=None,
            prox=flat_denoiser,
            shape=(1, 3 * height * width),
            cls=ProxFunc,
        )

        # Define the solver
        # solver = PGD(f=loss+op_score)
        solver = PGD(f=loss, g=op_denoiser)

        # Set the solver parameters
        # solver.fit(x0=x0.ravel(), tau=1e-5, acceleration=acceleration, stop_crit=stop_crit)
        solver.fit(x0=x0.ravel(), acceleration=acceleration, stop_crit=stop_crit)
        x0 = solver.solution()

        # Update plots
        recons = x0.reshape(3, height, width).transpose(1, 2, 0).mean(-1)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={"projection": sky_image.image_acc.wcs.sub([1, 2])})
        show_image(axs[0], x.mean(0), sky_im=sky_image, title="Dirty")
        show_image(axs[1], recons, sky_im=sky_image, title=f"Iteration {counter}")

        # Save the figure as an image
        plt.savefig(os.path.join(image_folder, f"step_{counter:04d}.png"))
        plt.close(fig)  # Close the figure to free up memory

        counter += 1

    # Create a video from the images
    images = [
        f"step_{i*frame_step:04d}.png"
        for i in range(len(os.listdir(image_folder)))
        if f"step_{i*frame_step:04d}.png" in os.listdir(image_folder)
    ]

    with imageio.get_writer(f"results/reconstruction.gif", mode="I", duration=100) as writer:
        for filename in images:
            image_frame = imageio.v3.imread(os.path.join(image_folder, filename))
            writer.append_data(image_frame)

print("Done!")