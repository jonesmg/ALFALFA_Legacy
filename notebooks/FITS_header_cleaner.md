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
# record header cleaner history
header_cleaner_version = 1.3
apply_version_list = [1.1, 1.2, 1.3]
```

This notebook takes an ALFALFA grid (as hosted in the NRAO archive) and updates the header keywords. This ensures the headers are accurate and compatible with modern standards, e.g., enabling spectral axis conversions in CARTA.

**Note**: You will need to have already downloaded the grid you wish to use and placed it in the same directory as this notebook. You can find instructions for accessing the grids in the [grid_access.md](../docs/grid_access.md) file in the docs folder and illustrated instructions on the [wiki](https://github.com/ALFALFAsurvey/ALFALFA_Legacy/wiki/ALFALFA-Grid-Access). In this case you need the 1044+13 grid and the "a" spectral cube needs to be placed in the current working directory.


changelog 
- 1.3
    - correct spectral by setting CRPIX3 to 1
    - change to not use radio velocity for VELREF
- 1.2
    - change celestial coordinate reference pixels
    - change pixel size
- 1.1
    - modify celestial coordinate keywords
    - modify frequency keywords, correct rest-frame frequency
    - correct polarization parameters
    - add beam polarization angle
    - correct unit parameters


# Imports

```python
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
```

# Read in data

```python
# Define the grid you are using
grid_ra = '1044'
grid_dec = '13'
freq_slice = 'a'

grid_filename = f'{grid_ra}+{grid_dec}{freq_slice}_spectral.fits'

# open the grid
cube_use =  fits.open(grid_filename)
```

```python
# open the header
header_new = cube_use["PRIMARY"].header.copy()
```

```python
# view the current header
header_new
```

Check the header cleaner version, and decide what modifications to apply. Can be changed manually

```python
if "Fits header cleaner" in header_new["HISTORY"]:
    print("Find previous header modification, will skip for version:")
    for item in header_new["history"][list(header_new["history"]).index("Fits header cleaner"):]:
        if "FHC: version " in item:
            hist_version = float(item.strip("FHC: version "))
            if hist_version in apply_version_list:
                print(hist_version)
                apply_version_list.remove(hist_version)
```

```python
print("Will apply header cleaner version: %s" % apply_version_list)
```

# version 1.1: correcting keywords and parameters


Change the celestial coordinate keywords to conform with the fits standard

```python
if 1.1 in apply_version_list:
    header_new.insert('CRPIX1', ("CUNIT1", "deg", ), after=True)
    header_new.insert('CRPIX2', ("CUNIT2", "deg", ), after=True)
    header_new["CTYPE1"] = "RA---TAN"
    header_new["CTYPE2"] = "DEC--TAN"
    header_new["CDELT1"] = -header_new["CDELT2"]
    header_new.insert('INSTRUME', ("LONPOLE", 180.0, ), after=True)  # necessary to conform with fits standard
```

Change the keywords for the frequency axis, and correct the rest-frame frequency.

```python
if 1.1 in apply_version_list:
    header_new["CTYPE3"]  = "FREQ"
    header_new.insert('CRPIX3', ("CUNIT3", "MHz", ), after=True)
    header_new["CRPIX3"] = 1
    header_new.insert('CUNIT3', ("CNAME3", "FREQ-HEL", ), after=True)
    
    header_new.insert('EQUINOX', ("SPECSYS", "HELIOCEN", "Spectral reference frame"), after=True)
    header_new.remove("EPOCH")
    header_new["VELREF"] = (2, "1 LSR, 2 HEL, 3 OBS, +256 Radio")
    
    header_new.rename_keyword("RESTFREQ", "RESTFRQ")  # rest frequency key word should be RESTFRQ
    header_new["RESTFRQ"] = (1420.405751e6, "Rest-frame frequency (Hz)")
```

Change the stokes parameters to LL and RR

```python
if 1.1 in apply_version_list:
    header_new["CRVAL4"] = -2  # LL and RR
```

Add keywords for beam

```python
if 1.1 in apply_version_list:
    header_new.insert('BMIN', ("BPA", 0, "ALFALFA beam position angle"), after=True)
```

Change BUNIT so it can be parsed by astropy.unit

```python
if 1.1 in apply_version_list:
    header_new.insert('BUNIT', ("BTYPE", "Intensity", ))
    header_new["BUNIT"] = "mJy/beam"
```

```python
if 1.1 in apply_version_list:
    if "Fits header cleaner" not in header_new["HISTORY"]:
        header_new.add_history("Fits header cleaner")
    header_new.add_history("FHC: version 1.1")
```

# version 1.2: reset celestial coordinate


Set the reference pixel to the central pixel, at the position marked by the grid name

```python
if 1.2 in apply_version_list:
    grid_ra, grid_dec = header_new["OBJECT"].split("+")
    center_pos = SkyCoord("%s:%s:00 %s:00:00" % (grid_ra[:2], grid_ra[2:], grid_dec), unit="hour, deg")
```

```python
if 1.2 in apply_version_list:
    header_new["CRVAL1"] = center_pos.ra.deg
    header_new["CRPIX1"] = header_new["NAXIS1"]/2. + 0.5
    header_new["CRVAL2"] = center_pos.dec.deg
    header_new["CRPIX2"] = header_new["NAXIS2"]/2. + 0.5
```

Change pixel size, each grid file should be 2.4 x 2.4 degree sampled by 144x144 pixels, so each pixel is exactly 1 arcmin

```python
if 1.2 in apply_version_list:
    header_new["CDELT1"] = -1./60  # degree
    header_new["CDELT2"] = 1./60
```

```python
if 1.2 in apply_version_list:
    if "Fits header cleaner" not in header_new["HISTORY"]:
        header_new.add_history("Fits header cleaner")
    header_new.add_history("FHC: version 1.2")
```

# Version 1.3: correct spectral axis


Fix the offset in the spectral axis. When converted to **optical velocity**, ...a grids should span 3293.53 to -2000.199 km/s.

```python
if 1.3 in apply_version_list:
    header_new["CRPIX3"] = 1
    header_new["VELREF"] = (2, "1 LSR, 2 HEL, 3 OBS, +256 Radio")
```

```python
if 1.3 in apply_version_list:
    if "Fits header cleaner" not in header_new["HISTORY"]:
        header_new.add_history("Fits header cleaner")
    header_new.add_history("FHC: version 1.3")
```

# Grouping into function

```python
def fits_header_clean(header, apply_version=None):
    
    header_new = copy.deepcopy(header)
    
    # version 1.2
    apply_version_list = apply_version if apply_version is not None else [1.1, 1.2, 1.3]
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
        header_new.insert('BMIN', ("BPA", 0, "ALFALFA beam position angle"), after=True)

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
        
    if 1.3 in apply_version_list:
        header_new["CRPIX3"] = 1
        header_new["VELREF"] = (2, "1 LSR, 2 HEL, 3 OBS, +256 Radio")
        if "Fits header cleaner" not in header_new["HISTORY"]:
            header_new.add_history("Fits header cleaner")
        header_new.add_history("FHC: version 1.3")
    
    return header_new
```

# Write fits

```python
# attach the new header to the data
cube_use["PRIMARY"].header = header_new
```

```python
# define the output grid name
out_ext = 'new'
grid_out_filename = f'{grid_ra}+{grid_dec}{freq_slice}_spectral_{out_ext}.fits'

# write the new grid out
cube_use.writeto(grid_out_filename, overwrite=True)
```
