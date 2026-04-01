---
jupyter:
  jupytext:
    default_lexer: ipython3
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import os, sys, glob
import copy
import numpy as np
import urllib

from scipy.stats import median_abs_deviation as mad
from matplotlib import pyplot as plt, colors

from astropy import constants as const, units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table as Tb
from astropy.wcs import WCS
from astropy.convolution import convolve, convolve_fft, Gaussian1DKernel

from PIL import Image
```

# Functions

```python
def fits_header_clean(header, apply_version=None):
    
    header_new = copy.deepcopy(header)
    
    # version 1.2
    apply_version_list = apply_version if apply_version is not None else [1.1, 1.2]
    if "Fits header cleaner" in header_new["HISTORY"]:
        print("Find previous header modification, will skip for version:")
        for item in header_new["history"][list(header_new["history"]).index("Fits header cleaner"):]:
            if "FHC: version " in item:
                hist_version = float(item.strip("FHC: version "))
                if hist_version in apply_version_list:
                    print(hist_version)
                    apply_version_list.remove(hist_version)
    print("Will apply header cleaner version: %s" % apply_version_list)

    if 1.1 in apply_version_list:
        # celestial coordinate
        header_new.insert('CRPIX1', ("CUNIT1", "deg", ), after=True)
        header_new.insert('CRPIX2', ("CUNIT2", "deg", ), after=True)
        header_new["CTYPE1"] = "RA---TAN"
        header_new["CTYPE2"] = "DEC--TAN"
        header_new["CDELT1"] = -header_new["CDELT2"]
        header_new.insert('INSTRUME', ("LONPOLE", 180.0, ), after=True)  # necessary to conform with fits standard
        
        # spectral axis
        header_new["CTYPE3"]  = "FREQ"
        header_new.insert('CRPIX3', ("CUNIT3", "MHz", ), after=True)
        header_new["CRPIX3"] = 1
        header_new.insert('CUNIT3', ("CNAME3", "FREQ-HEL", ), after=True)
        
        header_new.insert('EQUINOX', ("SPECSYS", "HELIOCEN", "Spectral reference frame"), after=True)
        header_new.remove("EPOCH")
        header_new["VELREF"] = (2, "1 LSR, 2 HEL, 3 OBS, +256 Radio")
        
        header_new.rename_keyword("RESTFREQ", "RESTFRQ")  # rest frequency key word should be RESTFRQ
        header_new["RESTFRQ"] = (1420.405751e6, "Rest-frame frequency (Hz)")

        # polarization axis
        header_new["CRVAL4"] = -2  # LL and RR
    
        # beam information
        header_new.insert('BMIN', ("BPA", 0, "ALFALFA beam position angles"), after=True)

        # unit
        header_new.insert('BUNIT', ("BTYPE", "Intensity", ))
        header_new["BUNIT"] = "mJy/beam"

        if "Fits header cleaner" not in header_new["HISTORY"]:
            header_new.add_history("Fits header cleaner")
        header_new.add_history("FHC: version 1.1")

    if 1.2 in apply_version_list:
        grid_ra, grid_dec = header_new["OBJECT"].split("+")
        center_pos = SkyCoord("%s:%s:00 %s:00:00" % (grid_ra[:2], grid_ra[2:], grid_dec), unit="hour, deg")

        header_new["CRVAL1"] = center_pos.ra.deg
        header_new["CRPIX1"] = header_new["NAXIS1"]/2. + 0.5
        header_new["CRVAL2"] = center_pos.dec.deg
        header_new["CRPIX2"] = header_new["NAXIS2"]/2. + 0.5

        header_new["CDELT1"] = -1./60  # degree
        header_new["CDELT2"] = 1./60

        if "Fits header cleaner" not in header_new["HISTORY"]:
            header_new.add_history("Fits header cleaner")
        header_new.add_history("FHC: version 1.2")
    
    return header_new
```

```python
# Function to average the two polarizations together and return a fits HDU object with the same header
def avg_pol(hdu):
    new_data = np.nanmean(hdu.data, axis=0, keepdims=True)
    new_header = copy.deepcopy(hdu.header)
    new_header["NAXIS4"] = 1
    new_header["CRVAL4"] = 0
    
    new_hdu = fits.PrimaryHDU(data=new_data, header=new_header)
    
    return new_hdu
```

```python
def make_vel(header):
    wcs_use = WCS(header)
    freq_arr= wcs_use.spectral.array_index_to_world(range(wcs_use.spectral.array_shape[0]))
    
    rest_freq = header['RESTFRQ'] * u.Hz
    vel_arr = freq_arr.to(u.km / u.s, equivalencies=u.doppler_optical(rest_freq), 
                          doppler_rest=rest_freq, doppler_convention="radio")

    return vel_arr
```

```python
def mask_gal(hdu, gal_vel=500*u.km/u.s):
    # build vhel array
    vel_arr = make_vel(hdu.header)
    # compare with galactic velocity range
    gal_mask = abs(vel_arr) < gal_vel

    return gal_mask[None, :, None, None]
```

```python
def make_mask_arr(hdu, mask=None):
    mask_arr = np.ones(hdu.data.shape, dtype=float)
    if mask is not None:
        mask_arr[mask] = np.nan

    return mask_arr
```

```python
def mask_weight(weight_hdu, threshold=0.75):
    # normalize weight cube
    weight_map_max = np.nanmax(weight_hdu.data, axis=1, keepdims=True)
    weight_cube_norm = weight_hdu.data / weight_map_max
    # mask by weight values
    wt_mask = weight_cube_norm < threshold

    return wt_mask
```

```python
def map_noise(hdu, mask=None, method="mad"):
    """
    method: 'mad' using median_abs_deviation, or 'std' using standard deviation
    """
    # initialize mask array
    mask_arr = make_mask_arr(hdu, mask=mask)
    # compute noise map
    if "mad" in method:
        noise_map = mad(hdu.data * mask_arr, axis=1, nan_policy="omit", ) * 1.48
    elif "std" in method:
        noise_map = np.nanstd(hdu.data * mask_arr, axis=1, )
    else:
        raise RuntimeError("Unknown method value: %s." % method)
    # construct hdu
    noise_cube = np.expand_dims(noise_map, axis=1)
    new_header = hdu.header.copy()
    new_header["NAXIS3"] = 1
    new_hdu = fits.PrimaryHDU(data=noise_cube, header=hdu.header)

    return new_hdu
```

```python
def mask_rms(hdu, threshold=2*2.33*u.mJy/u.beam, use_map_noise=False, mask=None, method="mad"):
    """
    param use_map_noise: if True, map_noise will be called, and threshold * noise_map 
        will be the threshold
    param threshold: if map_noise is False, threshold value will be the threshold 
    param method: passed to map_noise
    """
    # initialize mask array
    mask_arr = make_mask_arr(hdu, mask=mask)
    # clip by 
    if use_map_noise:
        noise_map = map_noise(hdu, mask=mask, method=method)
        threshold *= noise_map.data * u.Unit(noise_map.header["BUNIT"])
    rms_mask = hdu.data * mask_arr * u.Unit(hdu.header["BUNIT"]) > threshold

    return rms_mask
```

```python
def kern_smooth(hdu, kern, mask=None, conv_kwargs={}):
    new_data = np.empty_like(hdu.data)
    conv_kwargs_use = {"normalize_kernel": True, 
                       "nan_treatment": "interpolate", 
                       "preserve_nan": True}
    conv_kwargs_use.update(conv_kwargs)
    if np.ndim(kern) > 3:
        kern_use = kern[0]
    else:
        kern_use = kern
    for pol_idx, pol_arr in enumerate(hdu.data):
        if mask is not None:
            if mask.shape[0] > pol_idx:
                mask_use = mask[0]
            else:
                mask_use = mask[pol_idx]
        else:
            mask_use = mask
        new_data[pol_idx] = convolve_fft(
            pol_arr, kernel=kern_use, mask=mask_use, **conv_kwargs)
    
    new_hdu = fits.PrimaryHDU(data=new_data, header=hdu.header)

    return new_hdu
```

```python
def make_psf(header, ch_fwhm=2, bmaj_scale=1, bmin_scale=1):
    freq_psf = Gaussian1DKernel(ch_fwhm/2.355)
    dec_psf = Gaussian1DKernel(abs(header["BMAJ"]*bmaj_scale/header["CDELT2"])/2.355)
    ra_psf = Gaussian1DKernel(abs(header["BMIN"]*bmin_scale/header["CDELT1"])/2.355)

    psf_arr = freq_psf.array[None, :, None, None] * \
                dec_psf.array[None, None, :, None] * \
                ra_psf.array[None, None, None, :]

    return psf_arr
```

```python
def map_mom(hdu, mom=0, mask=None, ):
    # initialize mask array
    mask_arr = make_mask_arr(hdu, mask=mask)

    vel_arr = make_vel(hdu.header)
    wt = hdu.data * u.Unit(hdu.header["BUNIT"]) * \
        abs(np.interp(vel_arr, vel_arr[1:]-vel_arr[:-1], np.diff(vel_arr)))[None, :, None, None] * mask_arr
    # stack
    if mom == 0:
        new_cube = wt
    elif mom == 1:
        mom0_map = map_mom(hdu, mom=0, mask=None, noise_map=None)
        new_cube = wt * vel_arr[None, :, None, None] / \
        (mom0_map.data * u.Unit(mom0_map.header["BUNIT"]))
    else:
        mom0_map = map_mom(hdu, mom=0, mask=None, noise_map=None)
        mom1_map = map_mom(hdu, mom=1, mask=None, noise_map=None)
        new_cube = wt * (vel_arr[None, :, None, None] - \
                         (mom1_map.data * u.Unit(mom1_map.header["BUNIT"]))) / \
            (mom0_map.data * u.Unit(mom0_map.header["BUNIT"]))
        
    new_data = np.nansum(new_cube, axis=1, keepdims=True)
    new_header = copy.deepcopy(hdu.header)
    new_header["NAXIS3"] = 1
    new_header["BUNIT"] = new_data.unit.to_string()
    mom_map = fits.PrimaryHDU(data=new_data.value, header=new_header)
    
    return mom_map
```

```python
def ppl_src_mask(hdu, ch_smooth_list=[0, 2, 4], rms_thre_list=[5, 8, 15], 
                 mask=None, method="mad", min_occur=2):
    src_mask = make_mask_arr(hdu, None) * 0
    
    for ch_smooth, rms_thre in zip(ch_smooth_list, rms_thre_list):
        ch_kern = Gaussian1DKernel(ch_smooth/2.355).array[None, :, None, None]
        conv_cube = kern_smooth(hdu, kern=ch_kern, mask=mask) if ch_smooth != 0 else hdu

        rms_mask = mask_rms(conv_cube, threshold=rms_thre, use_map_noise=True, 
                            mask=mask, method=method)

        # cross check pol
        if ch_smooth != 0:
            src_mask += kern_smooth(fits.PrimaryHDU(data=(rms_mask[0] & rms_mask[1])[None, :, :, :].astype(float), 
                                                    header=hdu.header), 
                                    kern=ch_kern/ch_kern.max(), mask=mask, conv_kwargs={"normalize_kernel": False}).data >= 1
        else:
            src_mask += (rms_mask[0] & rms_mask[1])

    src_mask = src_mask >= min(min_occur, len(ch_smooth_list))
    new_header = copy.deepcopy(hdu.header)
    new_header["NAXIS3"] = 1
    src_mask_hdu = fits.PrimaryHDU(data=src_mask.astype(float), header=hdu.header)
    # expand src mask by psf
    psf_smooth = make_psf(hdu.header, ch_fwhm=2, )
    src_exp_mask = kern_smooth(src_mask_hdu, psf_smooth, 
                               mask=mask, conv_kwargs={"normalize_kernel": False}).data
    src_exp_mask = src_exp_mask > 0.01

    return src_exp_mask
```

# Read in data

```python
# Read in A100 catalog
a100_tb = Tb.read("https://content.cld.iop.org/journals/0004-637X/861/1/49/revision1/apjaac956t2_mrt.txt", format="ascii.mrt")
```

```python
src_coord_list = SkyCoord(ra=["%ih%im%fs" % (rah, ram, ras) for (rah, ram, ras)
                              in zip(a100_tb["HIRAh"], a100_tb["HIRAm"], a100_tb["HIRAs"])], 
                          dec=["%s%id%im%fs" % (design, ded, dem, des) for (design, ded, dem, des)
                              in zip(a100_tb["HIDE-"], a100_tb["HIDEd"], a100_tb["HIDEm"], a100_tb["HIDEs"])], unit=("hour", "deg"))
```

Read in the grid file, will try to read in all fits (a-d) for the input RA and Dec. We demonstrate here the 1244+33 grid which contains the NGC 4631 group, known to display large scale tidal features (e.g. [Wang+23](https://ui.adsabs.harvard.edu/abs/2023ApJ...944..102W/abstract)).

```python
grid_path = "../data/A2010/pipeline.unknown_date/"
grid_ra = '1044'
grid_dec = '13'
```

```python
# Read in the data cube
grid_hdu_list, wgts_hdu_list = [], []
for suffix in ("a", "b", "c", "d"):
    fits_file = os.path.join(grid_path, '%s+%s%s_spectral.fits' % (grid_ra, grid_dec, suffix))
    if os.path.exists(fits_file):
        grid_hdu_list.append(fits.open(fits_file))
        wgts_hdu_list.append(fits.open(fits_file.replace("_spectral.fits", "_spectralweights.fits")))
    else:
        print(f'{fits_file} does not exist')
```

```python
# reference sky coordina
alfalfa_wcs = WCS(fits_header_clean(grid_hdu_list[0]["PRIMARY"].header)).celestial
```

```python
#Set the image size in pixels
n_pix = 1024

#Sets image size and pixel scale
x_wid = y_wid = n_pix
pixscale = np.sqrt(abs(np.prod(np.linalg.eig(alfalfa_wcs.celestial.pixel_scale_matrix).eigenvalues*
                                  alfalfa_wcs.celestial.pixel_shape)))/n_pix
#Sets the coordinates of image center
center_pos = alfalfa_wcs.celestial.array_index_to_world(
    np.median(range(alfalfa_wcs.celestial.pixel_shape[-1])), 
    np.median(range(alfalfa_wcs.celestial.pixel_shape[-2])))
```

```python
#Sets the DECaLS URL to pull both the fits and jpeg image from
fits_url = f"https://www.legacysurvey.org/viewer/cutout.fits?ra={center_pos.ra.deg}&dec={center_pos.dec.deg}&layer=ls-dr10&pixscale={pixscale*3600.}&width={x_wid}&height={y_wid}&bands=g"
fits_head = fits.getheader(fits_url)
DECaLS_url = f"https://www.legacysurvey.org/viewer/cutout.jpg?ra={center_pos.ra.deg}&dec={center_pos.dec.deg}&layer=ls-dr10&pixscale={pixscale*3600.}&width={x_wid}&height={y_wid}"

#Get WCS for image
DECaLS_projection = WCS(fits_head)

#Downloads and saves jpeg image
urllib.request.urlretrieve(DECaLS_url, f'{grid_ra}+{grid_dec}_DECaLS.jpeg')
```

# All grids


Running the multi-scale 3d source finding. Note that it might take 10-20 minutes depending on the computational power of your machine.

```python
mom0_all = None
for i in range(4):
    hdu_use = grid_hdu_list[i]["PRIMARY"]
    weight_hdu_use = wgts_hdu_list[i]["PRIMARY"]
    hdu_use.header = fits_header_clean(hdu_use.header)
    weight_hdu_use.header = fits_header_clean(weight_hdu_use.header)
    gal_mask = mask_gal(hdu_use)
    wt_mask = mask_weight(weight_hdu_use)
    src_mask = ppl_src_mask(hdu_use, 
                            ch_smooth_list=[0,  1, 2, 4, 8, 12, 24, 36, 48, 60, ], 
                            rms_thre_list= [3.5]*10, 
                            mask=gal_mask|wt_mask, min_occur=3)

    mom0_map = map_mom(hdu_use, mom=0, mask=gal_mask|wt_mask|~src_mask)

    if mom0_all is None:
        mom0_all = mom0_map
    else:
        mom0_all.data += mom0_map.data
```

```python
# convert ALFALFA table coordinates for image coordinates
src_xy = DECaLS_projection.world_to_pixel(src_coord_list)
use_flag = np.isfinite(src_xy[0]) & np.isfinite(src_xy[1]) & \
(src_xy[0] >= -0.5) & (src_xy[0] <= DECaLS_projection.pixel_shape[-1]-0.5) &\
(src_xy[1] >= -0.5) & (src_xy[1] <= DECaLS_projection.pixel_shape[-2]-0.5)
```

```python
#If you want the plot to display in a separate, interactive window uncomment the following command
#%matplotlib tk

#To change the plot display back to inline in the notebook uncomment the following command
#%matplotlib inline
```

```python
#Finally make the overlay

#Open the DECaLS jpeg that we downloaded
DECaLS_jpeg = Image.open(f'{grid_ra}+{grid_dec}_DECaLS.jpeg')

#Set the contour levels
min_contour = 2.355 * 2 * 5 # mJy km/s / beam
contour_levels = min_contour*2**np.arange(9)

#Make the plot
plt.figure(figsize=[8,8], dpi=200)
ax = plt.subplot(111,projection=DECaLS_projection)
ax.imshow(DECaLS_jpeg)
ax.contour(avg_pol(mom0_all).data[0, 0],colors=['w','yellow','orange','r','magenta', "skyblue", "cyan"],
           levels=contour_levels,linewidths=0.4,transform=ax.get_transform(alfalfa_wcs))

ax.scatter(src_xy[0][use_flag], src_xy[1][use_flag], marker="s", s=50, facecolor="none", edgecolor="w")

plt.xlabel('RA')
plt.ylabel('Dec')

plt.savefig('%s+%s_mom0.png' % (grid_ra, grid_dec, ))
plt.show()
```
