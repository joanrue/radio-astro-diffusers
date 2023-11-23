import pyxu.util as pxu
import pyxu.info.deps as pxd
# Importing necessary libraries and modules
import os

import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np

from pyxu.operator import NUFFT

os.environ["RASCIL_DATA"] = "../data/"

from rascil.processing_components import create_test_image
from ska_sdp_func_python.util.coordinate_support import skycoord_to_lmn
from ska_sdp_datamodels.configuration.config_create import (
    create_named_configuration,
)
from ska_sdp_datamodels.visibility import create_visibility
from ska_sdp_datamodels.science_data_model.polarisation_model import (
    PolarisationFrame,
)
from pyxu.operator import FFT, DiagonalOp

# Define a function to display the reconstructed images
def show_image(ax, image, sky_im, title):
    nd = pxd.NDArrayInfo.from_obj(image)
    # xp = nd.module()
    img = image.copy()
    # image = xp.clip(image, 0, 1)
    if nd.name == "CUPY":
        img = img.get()

    vmax = np.max(img)
    vmin = np.min(img)
    cm = ax.imshow(img, origin="lower", cmap="Greys", vmax=vmax, vmin=vmin)

    ax.set_xlabel(sky_im.image_acc.wcs.wcs.ctype[0])
    ax.set_ylabel(sky_im.image_acc.wcs.wcs.ctype[1])
    ax.set_title(title)
    plt.colorbar(cm, ax=ax, orientation="vertical", shrink=0.7)
    ax.set_title(title)

def get_direction_cosines(image):
    _, _, ny, nx = image["pixels"].shape
    lmesh, mmesh = np.meshgrid(np.arange(ny), np.arange(nx))
    ra_grid, dec_grid = image.image_acc.wcs.sub([1, 2]).wcs_pix2world(lmesh, mmesh, 0)
    ra_grid = np.deg2rad(ra_grid)
    dec_grid = np.deg2rad(dec_grid)
    directions = SkyCoord(
        ra=ra_grid.ravel() * u.rad,
        dec=dec_grid.ravel() * u.rad,
        frame="icrs",
        equinox="J2000",
    )
    l, m, _ = skycoord_to_lmn(directions, image.image_acc.phasecentre)
    jacobian = np.sqrt(1 - l ** 2 - m ** 2)
    direction_cosines = np.stack([l, m, jacobian - 1.0], axis=-1).reshape(-1, 3)
    return direction_cosines, jacobian



def create_dataset(name="LOWBD2",
             rmax=400.0,
             ra=+15.0 * u.deg,
             dec=-45.0 * u.deg):
    lowr3 = create_named_configuration(name=name, rmax=rmax)
    visibility = create_visibility(
        config=lowr3,
        times=np.linspace(-3, 3, 4) * np.pi / 12,
        frequency=np.r_[15e7],
        channel_bandwidth=np.r_[5e4],
        weight=1.0,
        phasecentre=SkyCoord(
            ra=ra, dec=dec, frame="icrs", equinox="J2000"
),
        polarisation_frame=PolarisationFrame("stokesI"),
        times_are_ha=False,
    )

    sky_image = create_test_image(
        phasecentre=visibility.phasecentre, frequency=np.r_[15e7], cellsize=2.5e-4
    )
    return sky_image, visibility


def create_forward(visibilities, xp):
    uvw = visibilities.visibility_acc.uvw_lambda.reshape(-1, 3)

    uvw -= uvw.min()
    uvw /= uvw.max()
    uvw = (uvw * 511).astype(int)
    mask = np.zeros((3, 512, 512, 2), dtype="float32")

    for pos in uvw:
        mask[:, pos[0], pos[1]] = True

    mask = np.fft.fftshift(mask, axes=(1, 2))
    mask = xp.array(mask)
    forward = (
        DiagonalOp(vec=mask.ravel()) * FFT(
        arg_shape=(3, 512, 512),
        axes=(1, 2),
        real=True
    )
    )
    # forward = FFT(
    #     arg_shape=(3, 512, 512),
    #     axes=(1, 2),
    #     real=True)


    return (1. / 512.) * forward

def add_noise(y, snr_db):
    """
    Adds Gaussian noise to a signal with a given SNR in dB.

    Args:
    y (numpy array): The input signal.
    snr_db (float): The desired SNR in dB.

    Returns:
    numpy array: The signal with added Gaussian noise.
    """
    xp = pxu.get_array_module(y)
    # Calculate the power of the signal
    P_signal = xp.mean(y**2)

    # Calculate the power of the noise needed to achieve the desired SNR
    P_noise = P_signal * 10 ** (-snr_db / 10)

    # Calculate the standard deviation of the noise
    noise_std = xp.sqrt(P_noise)

    # Generate the Gaussian noise
    noise = xp.random.normal(0, noise_std, y.shape).astype(y.dtype)

    # Add the noise to the signal
    y_noisy = y + noise

    return y_noisy, noise_std