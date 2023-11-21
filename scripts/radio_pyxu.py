# Importing necessary libraries and modules
import os

import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np

from pyxu.operator import NUFFT, L21Norm, Gradient, SquaredL2Norm, PositiveOrthant, PositiveL1Norm
from pyxu.opt.solver import PD3O
from pyxu.util import view_as_complex
from pyxu.opt.stop import RelError, MaxIter



os.environ["RASCIL_DATA"] = "./data/"

from rascil.processing_components import show_image, create_test_image, \
    plot_uvcoverage, plot_visibility
from ska_sdp_func_python.util.coordinate_support import skycoord_to_lmn
from ska_sdp_datamodels.configuration.config_create import (
        create_named_configuration,
    )
from ska_sdp_datamodels.visibility import create_visibility
from ska_sdp_datamodels.science_data_model.polarisation_model import (
        PolarisationFrame,
    )
    

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
    jacobian = np.sqrt(1 - l**2 - m**2)
    direction_cosines = np.stack([l, m, jacobian - 1.0], axis=-1).reshape(-1, 3)
    return direction_cosines, jacobian

from rascil.processing_components import show_image, create_test_image, \
    plot_uvcoverage, plot_visibility

lowr3 = create_named_configuration("LOWBD2", rmax=3_000.0)
vis = create_visibility(
    config=lowr3,
    times=np.linspace(-4, 4, 9) * np.pi / 12,
    frequency=np.r_[15e7],
    channel_bandwidth=np.r_[5e4],
    weight=1.0,
    phasecentre=SkyCoord(
        ra=+15.0 * u.deg, dec=-45.0 * u.deg, frame="icrs", equinox="J2000"
    ),
    polarisation_frame=PolarisationFrame("stokesI"),
    times_are_ha=False,
)

m31image = create_test_image(
        phasecentre=vis.phasecentre, frequency=np.r_[15e7], cellsize=5e-4
)
direction_cosines, jacobian = get_direction_cosines(m31image)
wgt_dirty = 1 / jacobian.reshape(-1)
# Show image

# Creating the NUFFT operator
uvw = vis.visibility_acc.uvw_lambda.reshape(-1, 3)
xyz = direction_cosines.reshape(-1, 3)                              
forward = NUFFT.type3(
    x=xyz, z=2 * np.pi * uvw, real=True, isign=-1,
)

x = m31image.pixels.data.squeeze().copy()
y = forward(x.ravel())
breakpoint()
adj = forward.adjoint(y)
print(any(np.isnan(adj)))
