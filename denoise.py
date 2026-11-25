"""
This script denoises the CT datasets passed in 'fnames' (.h5 files) using skimage.restoration denoise_nl_means (Nonlocal means). Only the index of the passed signal variable 
('attenuation', 'fused', 'refraction', 'diffusion') is loaded. This choice is due to the skimage implementation uses no parallelization and therefore
separate jobs for each signalVariable are run on a SLURM cluster. The patch size is hardcoded to 5 x 5 x 5 to balance denoising and computational
burden. The output is saved as a pickle file in this prototype for subsequent consumption.
"""

import datetime
import numpy as np
import xarray as xr
from skimage.restoration import denoise_nl_means, estimate_sigma
import pickle
import pandas as pd

signalVars = ['attenuation', 'fused', 'refraction', 'diffusion']

def load_val_subset(fname, data_vars, subset={}):
    # Open dataset lazily without loading it into memory
    ds = xr.open_dataset(fname)
    subset = ds[data_vars].isel(**subset)
    
    # Load only the selected data into memory
    d = subset.load()
    ds.close()  # Now safe to open another file
    return d
def denoise(fnames: list[str], s: str, measurements: dict[str, list[str]], bbox: dict, pressures: list[float]) -> None:
    data = xr.concat(
        [load_val_subset(fname, s, bbox) for fname in fnames],
        pd.Index(data=pressures, name = "dP"),
        join='exact'
    )

    denoised = xr.DataArray(
            np.empty((len(data.dP), len(data.z), len(data.y), len(data.x))),
            coords=[data.dP, data.z, data.y, data.x],
            dims=["dP", "z", "y", "x"]
        )

    patch_kw = dict(
        patch_size=5,  # 5x5x5 patches
        patch_distance=10,  # 21x21x21 search area
        channel_axis=None, # greyscale image
    )

    for i, p in enumerate(pressures):
        volume = data.sel(dP=p).values
        sigma_est = np.mean(estimate_sigma(volume, channel_axis=None))
        print(datetime.datetime.now())
        print(f'estimated noise standard deviation = {sigma_est} for {p} kPa')
        denoised.loc[{'dP': p}] = denoise_nl_means(volume, h=0.9 * sigma_est, sigma=sigma_est, fast_mode=True, **patch_kw)

    dataset = xr.Dataset(data_vars={f'{s}': data, f'{s}_denoised': denoised})
    output = open(f'{s}_denoised.pkl', 'wb')
    pickle.dump(dataset, output)
    output.close()