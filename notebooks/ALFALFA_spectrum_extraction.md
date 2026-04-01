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
import os, sys
import numpy, astropy
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.table import Table
import astropy.units as u
import matplotlib.pyplot as plt
from photutils.aperture import SkyEllipticalAperture, EllipticalAperture, SkyRectangularAperture, RectangularAperture 

#Add the scripts directory path
scripts_path = os.path.abspath("../scripts")
sys.path.append(scripts_path)

from load_grid import load_grid
```

This notebook takes an ALFALFA cube and extract a 1-dimensional spectrum at a specified location. It does this first over an area equal to the beam shape and then over a larger square region.

**Note**: You will need the ALFALFA 1044+13 grid in that data directory in order for this notebook to run successfully. You can obtain this by running the [download_example_data.py](../scripts/download_example_data.py) script in the scripts directory. If you wish to run this with other grids then you can find instructions for accessing the grids in the [grid_access.md](../docs/grid_access.md) file in the docs folder or illustrated instructions on the [wiki](https://github.com/ALFALFAsurvey/ALFALFA_Legacy/wiki/ALFALFA-Grid-Access).

```python
#Define the path to the data
#You should already have run the data download script
cwd = os.getcwd()+'/'
data_path = cwd+'../data/A2010/pipeline.unknown_date/'
```

```python
#Define the grid you are using
grid_ra = '1044'
grid_dec = '13'
freq_slice = 'b'

#Use the load_grid function to load the cube and WCS
cube, freq, vel, grid_wcs, header = load_grid(data_path,grid_ra,grid_dec,freq_slice)

#Average together the two polarizations
cube = numpy.mean(cube,axis=0)

#Extract beam parameters
bmaj, bmin, bpa = header['BMAJ']*u.deg, header['BMIN']*u.deg, 0*u.deg
pixsize = header['CDELT2']*u.deg

#Make the beam correction factor
beam_factor = (numpy.pi*bmaj*bmin/(pixsize**2.))/(4.*numpy.log(2.))

#Extract the channel width from the header and build frequency array
chan_df = header['CDELT3']*u.MHz
```

```python
#Start by extracting a spectrum in the shape of the beam

#Define a position to extract a spectrum at
#In this case we will use AGC200878
spec_pos = SkyCoord(160.739480, 13.548581, unit=u.deg)

#Make an aperture that's at least twice the size of the beam
sky_aperture = SkyEllipticalAperture(spec_pos, a=bmaj, b=bmin, theta=bpa)

#Convert to a pixel aperture
pix_aperture = sky_aperture.to_pixel(grid_wcs.celestial)
```

```python
#Plot the aperture on the grid

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(projection=grid_wcs.celestial)
plt.imshow(numpy.max(cube,axis=0), vmin=0, vmax=40,
           origin='lower', cmap='Greys', aspect='equal')

pix_aperture.plot(color='orange',lw=1.5,ls='-')

plt.xlabel(r'RA')
plt.ylabel(r'Dec')
```

```python
#Now extract the spectrum
pix_aperture_mask = pix_aperture.to_mask(method='exact')

#Create empty spectrum
spec = numpy.zeros(len(freq))

#Step through cube channels and extract flux in each
for i in range(len(spec)):
    spec[i] = numpy.sum(pix_aperture_mask.multiply(cube[i]))/beam_factor

#Set units
spec = spec*u.mJy
```

```python
plt.plot(vel,spec)

plt.axhline(0.,c='k',lw=1)
plt.ylim(-10.,25.)

plt.xlabel('Velocity [km/s] (Heliocentric)')
plt.ylabel('Flux Density [mJy]')
```

We can compare the extracted spectrum to the published ALFALFA spectrum from a.100 (Haynes et al. 2018).

```python
alf_spec = Table.read("https://vizier.cds.unistra.fr/viz-bin/ftp-index?J/ApJ/861/49/sp/A200878.fits")
```

```python
plt.plot(vel,spec, label="Extracted", c='tab:blue')
plt.plot(alf_spec["VHELIO"], alf_spec["FLUX"], label="a.100", c='tab:orange', alpha=0.75)

plt.axhline(0.,c='k',lw=1)
plt.ylim(-10.,25.)

plt.xlabel('Velocity [km/s] (Heliocentric)')
plt.ylabel('Flux Density [mJy]')

plt.legend()
```

The extracted spectrum matches closely with the published ALFALFA spectrum. However, for more extended sourced the situation can be more complex.


To illustrate this we will now extract the spectrum of the nearby galaxy UGC5826. This is a spatially extended source, so we will extract it over a 20'x20' square region. We also need to load the 'a' frequency slice as this is a lower velocity source.

```python
#Define the grid you are using
grid_ra = '1044'
grid_dec = '13'
freq_slice = 'a'

#Use the load_grid function to load the cube and WCS
cube, freq, vel, grid_wcs, header = load_grid(data_path,grid_ra,grid_dec,freq_slice)

#Average together the two polarizations
cube = numpy.mean(cube,axis=0)

#Extract beam parameters
bmaj, bmin, bpa = header['BMAJ']*u.deg, header['BMIN']*u.deg, 0*u.deg
pixsize = header['CDELT2']*u.deg

#Make the beam correction factor
beam_factor = (numpy.pi*bmaj*bmin/(pixsize**2.))/(4.*numpy.log(2.))

#Extract the channel width from the header and build frequency array
chan_df = header['CDELT3']*u.MHz
```

```python
#Start by extracting a spectrum in the shape of the beam

#Define a position to extract a spectrum at
#In this case we will use UGC5826
spec_pos = SkyCoord(160.531332, 13.747050, unit=u.deg)

#Make an aperture that's at least twice the size of the beam
sky_aperture = SkyEllipticalAperture(spec_pos, a=bmaj, b=bmin, theta=bpa)

#Convert to a pixel aperture
pix_aperture = sky_aperture.to_pixel(grid_wcs.celestial)
```

```python
#Make a rectangular aperture that's 20x20 arcmin
sky_aperture = SkyRectangularAperture(spec_pos,w=20*u.arcmin,h=20*u.arcmin)

#Convert to a pixel aperture
pix_aperture = sky_aperture.to_pixel(grid_wcs.celestial)
```

```python
#Plot the aperture on the grid

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(projection=grid_wcs.celestial)
plt.imshow(numpy.sum(cube,axis=0),origin='lower', cmap='Greys', aspect='equal')

pix_aperture.plot(color='orange',lw=1.5,ls='-')

plt.xlabel(r'RA')
plt.ylabel(r'Dec')
```

```python
#Now extract the spectrum
pix_aperture_mask = pix_aperture.to_mask(method='exact')

#Create empty spectrum
spec = numpy.zeros(len(freq))

#Step through cube channels and extract flux in each
for i in range(len(spec)):
    spec[i] = numpy.sum(pix_aperture_mask.multiply(cube[i]))/beam_factor

#Set units
spec = spec*u.mJy
```

```python
plt.plot(vel,spec)

plt.axhline(0.,c='k',lw=1)
plt.ylim(-50.,1200.)

plt.xlabel('Velocity [km/s] (Heliocentric)')
plt.ylabel('Flux Density [mJy]')
```

Now let's compare with the spectrum catalogued in ALFALFA and a spectrum from the Green Bank 140ft telescope (Springob et al. 2005).

```python
alf_spec = Table.read("https://vizier.cds.unistra.fr/viz-bin/ftp-index?J/ApJ/861/49/sp/A005826.fits")

gbt_spec = Table.read("https://ned.ipac.caltech.edu/spc1/2005/2005ApJS..160..149S/UGC_05826:S:HI:shg2005.txt",
                      format='ascii.commented_header', names=['vel','freq','flux','base'])
```

```python
plt.plot(vel,spec, label="Extracted", c='tab:blue')
plt.plot(gbt_spec["vel"], gbt_spec["flux"]-gbt_spec["base"], label="GB140ft", c='tab:green')
plt.plot(alf_spec["VHELIO"], alf_spec["FLUX"], label="a.100", c='tab:orange')

plt.axhline(0.,c='k',lw=1)
plt.ylim(-50.,1200.)

plt.xlabel('Velocity [km/s] (Heliocentric)')
plt.ylabel('Flux Density [mJy]')

plt.legend()
```

<!-- #region -->
The a.100 and the GB 140ft spectrum agree very well, but the extracted spectrum appears to contains significantly more flux, however, some of this additional flux may have been double counted due to the sidelobes of Arecibo.

Users are reminded that the extraction of fluxes for extended sources from the ALFALFA grids is complicated by the nature of the data products as constructs from transit drift scans made by the seven-horn ALFA front-end array and the significant sidelobes of the Arecibo antenna. These effects have been discussed in Irwin et al. (2009), Dowell (2010) and Hoffman et al. (2019). 

Because the main objective of ALFALFA was to obtain a complete census for the dominant population of sources significantly smaller than the ~3.5 arcmin beam, the grid production and source extraction process was optimized for those sources. Although some attempt was made to measure fluxes over an extended area when a source was clearly resolved, the fluxes in the a100 catalog (Haynes et al. 2018) for extended sources carry a large uncertainty.

**Users should be wary of possible beam effects when using these grids for sources comparable to or larger than the native ALFA beam (3.3 x 3.8 arcmin).**


References:

Springob et al. (2005), https://ui.adsabs.harvard.edu/abs/2005ApJS..160..149S/abstract

Irwin et al. (2009), https://ui.adsabs.harvard.edu/abs/2009ApJ...692.1447I/abstract

Dowell 2010, PhDThesis, https://ui.adsabs.harvard.edu/abs/2010PhDT.......102D/abstract

Haynes et al. (2018), https://ui.adsabs.harvard.edu/abs/2018ApJ...861...49H/abstract

Hoffman et al. (2019), https://ui.adsabs.harvard.edu/abs/2019AJ....157..194H/abstract
<!-- #endregion -->
