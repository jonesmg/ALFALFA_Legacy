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
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import urllib
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import requests
from PIL import Image

#Add the scripts directory path
scripts_path = os.path.abspath("../scripts")
sys.path.append(scripts_path)

from load_grid import load_grid, load_cont_grid
```

This notebook takes an ALFALFA grid file, makes a quick moment zero map, downloads a DECaLS image for the corresponding sky area, and finally overlays HI contours on the image.

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
freq_slice = 'a'

#Use the load_grid function to load the cube and WCS
cube, freq, vel, grid_wcs, header = load_grid(data_path,grid_ra,grid_dec,freq_slice)
```

```python
#Average together the two polarizations
cube = numpy.mean(cube,axis=0)
```

```python
#Identify a line-free channel that is mostly just noise to estimate the rms from
chan = 300

#Calculate the rms
rms = numpy.sqrt(numpy.mean(numpy.square(cube[chan].flatten())))
print(f'RMS noise = {rms} mJy/beam')

#Plot the channel to make sure it doesn't contain sources
plt.imshow(cube[chan],origin='lower')
```

```python
#Make a simple moment zero map over a select channel range
min_chan = 0
max_chan = 550

#Blank regions of the cube below 3 x rms
blanked_cube = numpy.where(cube[min_chan:max_chan] > 3*rms, cube[min_chan:max_chan], 0.)

#Sum the blanked cube to make the moment zero map
mom0 = numpy.sum(blanked_cube, axis = 0)

#Quickly plot the moment map
plt.imshow(mom0,origin='lower')
```

<br>
<br>

To give the moment zero map more physical units we should also multiply it by the width of a channel. This could be done simply in frequency to give units of Jy Hz / beam, but we will instead make the units Jy km/s / beam.

```python
#Find the average channel width over the range used for the moment map
#Technically you should do this channel by channel, but this is close enough
chan_dv = numpy.mean(vel[min_chan:max_chan-1]-vel[min_chan+1:max_chan])
print(f'Channel width = {chan_dv}')
```

```python
#Now multiply the moment zero map by the channel width and divide by 1000 to get Jy.km/s
mom0_Jykms = mom0*chan_dv.value/1000
```

<br>

Now we need to decide on the dimensions that we want for our DECaLS image, build a WCS for it, and download the image. Then we need to reproject the moment zero map to the same dimensions as the DECaLS image and overlay them.

```python
#Set the image size in pixels
n_pix = 1024

#Sets image size and pixel scale
x_wid, y_wid = n_pix, n_pix
pixscale = abs(grid_wcs.wcs.cdelt[1])*144/n_pix

#Sets the coordinates of image center
center_ra = (numpy.floor(float(grid_ra)/100) + (float(grid_ra)-numpy.floor(float(grid_ra)/100)*100)/60)*15
center_dec = grid_dec
center_pos = SkyCoord(center_ra,center_dec,unit='deg')
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

```python
plt.imshow(mom0_Jykms,origin='lower')
plt.colorbar()
```

```python
#Finally make the overlay

#Open the DECaLS jpeg that we downloaded
DECaLS_jpeg = Image.open(f'{grid_ra}+{grid_dec}_DECaLS.jpeg')

#Set the contour levels
min_contour = 0.5 #Jy km/s / beam
contour_levels = min_contour*numpy.array([1,2,4,8,16])

#Make the plot
plt.figure(figsize=[8,8])
ax = plt.subplot(111,projection=DECaLS_projection)
ax.imshow(DECaLS_jpeg)
ax.contour(mom0_Jykms,colors=['w','yellow','orange','r','magenta'],levels=contour_levels,linewidths=2,
           transform=ax.get_transform(grid_wcs.celestial))
plt.xlabel('RA')
plt.ylabel('Dec')
```

In the original example grid (1044+13a) there is a clear artifact near the center of the FoV. This can easily be removed by using the accompanying weights cube. Note that the weighting has already been applied to the flux scale in the data cube and that the numerical scale of the weights is aritrary. Thus, the weights should be used to scale/weight the data as this will make the flux scale non-physical. However, they can be used to mask regions of the data.

```python
#Use the load_grid function again, but now also load the weights cube
cube, freq, vel, grid_wcs, header, wgt_cube, wgt_header = load_grid(data_path,'1044','13','a',include_weights=True)
```

```python
#Plot an example of the weight
plt.imshow(wgt_cube[0][200],origin='lower')
```

```python
#Calculate the maximum weight value
wgt_max = max(wgt_cube.flatten())
print(wgt_max)

#Blank regions with low weights
#Here we have set the minimum weight to 30%, but this may need to be adjusted for other grids
blanked_cube = numpy.where(wgt_cube/wgt_max > 0.3, cube, 0.)

#Average together the two polarizations
blanked_cube = numpy.mean(blanked_cube,axis=0)
cube = numpy.mean(cube,axis=0)

#Blank regions of the cube below 3 x rms
blanked_cube = numpy.where(blanked_cube[min_chan:max_chan] > 3*rms, 
                           cube[min_chan:max_chan], 0.)

#Sum the blanked cube to make the moment zero map
mom0 = numpy.sum(blanked_cube, axis = 0)

#Change scale to Jy / km/s / beam
mom0_Jykms = mom0*chan_dv.value/1000
```

```python
#If you would like the plot to appear in a seperate, interactive window please uncomment the line below.
#%matplotlib tk

#To return the plotting style to displaying within the notebook, run the following command:
#%matplotlib inline
```

```python
#Finally make the overlay

#Open the DECaLS jpeg that we downloaded
DECaLS_jpeg = Image.open(f'{grid_ra}+{grid_dec}_DECaLS.jpeg')

#Set the contour levels
min_contour = 0.5 #Jy km/s / beam
contour_levels = min_contour*numpy.array([1,2,4,8,16])

#Make the plot
plt.figure(figsize=[8,8])
ax = plt.subplot(111,projection=DECaLS_projection)
ax.imshow(DECaLS_jpeg)
ax.contour(mom0_Jykms,colors=['w','yellow','orange','r','magenta'],levels=contour_levels,linewidths=2,
           transform=ax.get_transform(grid_wcs.celestial))
plt.xlabel('RA')
plt.ylabel('Dec')
```

You can also load and display the ALFALFA continuum grids in a similar way, although, of course, it isn't necessary to construct the moment zero map for the continuum.

```python
#Define the grid you are using
grid_ra = '1044'
grid_dec = '13'
freq_slice = 'a'

#Use the load_cont_grid function to load the continuum map and WCS
cont, grid_wcs, header = load_cont_grid(data_path,grid_ra,grid_dec,freq_slice)
```

```python
#Average the two polarizations
cont = numpy.mean(cont,axis=0)
```

```python
#Now make conthe overlay

#Open the DECaLS jpeg that we downloaded
DECaLS_jpeg = Image.open(f'{grid_ra}+{grid_dec}_DECaLS.jpeg')

#Set the contour levels
min_contour = 20 #Jy km/s / beam
contour_levels = min_contour*numpy.array([1,2,4,8,16])

#Make the plot
plt.figure(figsize=[8,8])
ax = plt.subplot(111,projection=DECaLS_projection)
ax.imshow(DECaLS_jpeg)
ax.contour(cont,colors=['w','yellow','orange','r','magenta'],levels=contour_levels,linewidths=2,
           transform=ax.get_transform(grid_wcs.celestial))
plt.xlabel('RA')
plt.ylabel('Dec')
```

Note that although there are four continuum grids at each position (a, b, c, d) to match the four velocity ranges of the line cubes, the data in these four continuum files are identical, that is, each contains the continuum emission for the entire bandpass.
