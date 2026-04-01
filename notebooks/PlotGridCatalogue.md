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
    display_name: Python [conda env:AALegacy]
    language: python
    name: conda-env-AALegacy-py
---

# All Arecibo Sky Plots

Plot both the grids, and the ALFALFA100 [Haynes et al. 2018](http://adsabs.harvard.edu/abs/2018ApJ...861...49H) sources.

```python
#Import statements

import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle

from astropy import wcs

from regions import RectangleSkyRegion
import numpy as np

import matplotlib.pyplot as plt

from astroquery.vizier import Vizier
```

```python
#Defining some functions

def makegridwcs(sky_center):
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [float(sky_center.ra/u.deg)+1, float(sky_center.dec/u.deg)+1]
    #Adding one makes pixel values identical to RA and Dec
    w.wcs.cdelt = [1.0,1.0]
    w.wcs.crval = [float(sky_center.ra/u.deg), float(sky_center.dec/u.deg)]
    w.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    return w

def plotgrid(i):
    sky_region = grids["box"][i]
    sky_center = grids["coord"][i]
    w = makegridwcs(sky_center)            
    pixel_region = sky_region.to_pixel(w)
    pixel_region.plot(ax=ax, color='red', lw=1.0)
```

```python
#Opening the list of grids, and putting it into a Pandas database

with open("gridlist_july2025.txt", "r") as f:
    grids = f.read().split()

grids = pd.DataFrame({"gridname": grids})
grids["coord"] = grids.gridname.map(lambda x: SkyCoord(x[0:2] + ":" + x[2:], unit=(u.hourangle, u.deg)))
grids["box"]   = grids.coord.map(lambda x:RectangleSkyRegion(x, width=2.4*u.deg, height = 2.4*u.deg))

grids["ra"] = grids["coord"].map(lambda x: round(float(x.ra/u.deg)))
grids["dec"] = grids["coord"].map(lambda x: round(float(x.dec/u.deg)))
```

```python
#Download the ALFALFA 100% from Vizier/CDS
cat = Vizier(catalog="J/ApJ/861/49/table2", columns=['*', '_RAJ2000', '_DEJ2000'], row_limit=-1).query_constraints()[0]
```

```python
cat
```

```python
#Plot the ALFALFA Spring Sky (RA = 07h30m to 16h30m or 112.5-247.5deg)

fig, ax = plt.subplots(figsize=(20,6))
plt.xlim(260,100)
plt.ylim(-2,37)
plt.plot(cat["_RAJ2000"],cat["_DEJ2000"],'ko', markersize=1)
plt.ylabel('Dec (deg)') 
plt.xlabel('RA (deg)')

for i in range(len(grids)): plotgrid(i)
```

```python
#Plot one half of the Fall sky
fig, ax = plt.subplots(figsize=(5,5))

plt.xlim(360, 320)
plt.plot(cat["_RAJ2000"],cat["_DEJ2000"],'ko', markersize=1)
plt.ylim(-2,37)
plt.ylabel('Dec (deg)') 
plt.xlabel('RA (deg)')

for i in range(len(grids)): plotgrid(i)
```

```python
#Plot the other half of the Fall sky
fig, ax = plt.subplots(figsize=(5,5))

plt.xlim(55, 0)
plt.plot(cat["_RAJ2000"],cat["_DEJ2000"],'ko', markersize=1)
plt.ylim(-2,37)
plt.ylabel('Dec (deg)') 
plt.xlabel('RA (deg)')

for i in range(len(grids)): plotgrid(i)
```

```python
# Plot the entire sky: 

fig, ax = plt.subplots(figsize=(20,3))
plt.xlim(365, -5)
plt.plot(cat["_RAJ2000"],cat["_DEJ2000"],'ko', markersize=1)
plt.ylim(-2,37)
plt.ylabel('Dec (deg)') 
plt.xlabel('RA (deg)')

for i in range(len(grids)): plotgrid(i)
```

```python

```
