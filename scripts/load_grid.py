import astropy.io.fits as fits
from astropy.wcs import WCS
import numpy as np
import astropy, copy
import astropy.units as u
from astropy.coordinates import SkyCoord

def fits_header_clean(header, grid_ra, grid_dec, apply_version=None):
    '''
    This function takes the ALFALFA grid headers, as orignally stored in NRAO,
    and corrects the keywords so that they follow modern FITS standards.

    INPUTS:
    header: FITS header object.
    grid_ra: (String) Right Ascension of the grid to be loaded, e.g., "1044".
    grid_dec: (String) Declination of the grid to be loaded, e.g., "13".
    apply_version: (Float) Version of the correct to apply. Defaults to latest.

    OUTPUTS:
    new_header: FITS header object.
    '''
    
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
        if header_new["NAXIS"] == 4:  # only if the spectral axis exists
            header_new["CTYPE3"]  = "FREQ"
            header_new.insert('CRPIX3', ("CUNIT3", "MHz", ), after=True)
            header_new["CRPIX3"] = 1
            header_new.insert('CUNIT3', ("CNAME3", "FREQ-HEL", ), after=True)
            
            header_new.insert('EQUINOX', ("SPECSYS", "HELIOCEN", "Spectral reference frame"), after=True)
            header_new.remove("EPOCH")
            header_new["VELREF"] = (2, "1 LSR, 2 HEL, 3 OBS, +256 Radio")

            # polarization axis
            header_new["CRVAL4"] = -2  # LL and RR
        else:
            # polarization axis
            header_new["CRVAL3"] = -2  # LL and RR
        
        header_new.rename_keyword("RESTFREQ", "RESTFRQ")  # rest frequency key word should be RESTFRQ
        header_new["RESTFRQ"] = (1420.405751e6, "Rest-frame frequency (Hz)")

        # beam information
        if "BMIN" in header_new:  # only modify if header contains beam information
            header_new.insert("BMIN", ("BPA", 0, "ALFALFA beam position angles"), after=True)

        # unit
        if "BUNIT" in header_new:  # only modify if header contains unit information
            header_new.insert("BUNIT", ("BTYPE", "Intensity", ))
            header_new["BUNIT"] = "mJy/beam"

        if "Fits header cleaner" not in header_new["HISTORY"]:
            header_new.add_history("Fits header cleaner")
        header_new.add_history("FHC: version 1.1")

    if 1.2 in apply_version_list:
        grid_ra, grid_dec = header_new["OBJECT"].split("+")
        center_pos = SkyCoord("%s:%s:00 %s:00:00" % (grid_ra[:2], grid_ra[2:], grid_dec), unit="hour, deg")
        #Note that the ALFALFA source positions were corrected as described in section 3.4.2
        #of Brian Kent's PhD thesis: https://ecommons.cornell.edu/items/a278d737-b863-4e07-b637-88587dff669f
        #However, these corrections were NOT applied to the grids themselves.
        #Here we construct an approximate WCS that should be suitable for most applications.

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

def load_grid(dir_path, grid_ra, grid_dec, freq_slice, include_weights=False):
    '''
    This function loads the requested ALFALFA grid and applies corrections to
    the header keywords and an approximate fix for the world coordinate system.
    The weights cubes is also loaded if requested.

    INPUTS:
    dir_path: (String) Path to the directory containing the data.
    grid_ra: (String) Right Ascension of the grid to be loaded, e.g., "1044".
    grid_dec: (String) Declination of the grid to be loaded, e.g., "13".
    freq_slice: (String) Frequency slice of the ALFALFA data, e.g., "a", "b", "c", or "d".
    include_weights: (Bool) Return weights cube as well as data (optional).

    OUTPUTS:
    cube: (Array) Data cube.
    freq: (Array) Frequency array for the cube (MHz).
    vel: (Array) Heliocentric velocity array for the cube (km/s).
    wcs: (WCS Object) World Coordinate System object for the grid (2-dimensional).
    header: (Header Object) FITS header of the cube.
    (Optional) weights_cube: (Array) Weights cube.
    (Optional) weights_header: (Header Object) FITS header of the weights cube.
    '''

    grid_filename = f'{dir_path}{grid_ra}+{grid_dec}{freq_slice}_spectral.fits'

    #Open the fits file
    hdu = fits.open(grid_filename)

    #Extract the data from fits file
    cube = hdu[0].data

    #Extract and correct header
    orig_header = hdu[0].header
    new_header = fits_header_clean(orig_header, grid_ra, grid_dec)

    #Start by building frequency and velocity arrays
    grid_wcs = WCS(new_header)
    freq = grid_wcs.spectral.array_index_to_world(range(grid_wcs.spectral.array_shape[0]))
    rest_freq = new_header['RESTFRQ'] * u.Hz
    vel = freq.to(u.km / u.s, equivalencies=u.doppler_optical(rest_freq),
                  doppler_rest=rest_freq, doppler_convention="optical")
    
    if include_weights:
        wgts_filename = f'{dir_path}{grid_ra}+{grid_dec}{freq_slice}_spectralweights.fits'
        
        #Open the fits file
        wgts_hdu = fits.open(wgts_filename)
    
        #Extract the data from fits file
        wgts_cube = wgts_hdu[0].data

        #Extract and correct header
        orig_wgts_header = wgts_hdu[0].header
        new_wgts_header = fits_header_clean(orig_wgts_header, grid_ra, grid_dec)

        return cube, freq, vel, grid_wcs, new_header, wgts_cube, new_wgts_header
    else:
        return cube, freq, vel, grid_wcs, new_header

def load_cont_grid(dir_path, grid_ra, grid_dec, freq_slice, include_weights=False):
    '''
    This function loads an ALFALFA continuum grid and applies corrections to
    the header keywords and an approximate fix for the world coordinate system.
    The weights cubes is also loaded if requested.

    INPUTS:
    dir_path: (String) Path to the directory containing the data.
    grid_ra: (String) Right Ascension of the grid to be loaded, e.g., "1044".
    grid_dec: (String) Declination of the grid to be loaded, e.g., "13".
    freq_slice: (String) Frequency slice of the ALFALFA data, e.g., "a", "b", "c", or "d".
    include_weights: (Bool) Return weights cube as well as data (optional).

    OUTPUTS:
    grid: (Array) Continuum grid.
    wcs: (WCS Object) World Coordinate System object for the grid (2-dimensional).
    header: (Header Object) FITS header of the cube.
    (Optional) weights_cube: (Array) Weights cube.
    (Optional) weights_header: (Header Object) FITS header of the weights cube.
    '''

    grid_filename = f'{dir_path}{grid_ra}+{grid_dec}{freq_slice}_continuum.fits'

    #Open the fits file
    hdu = fits.open(grid_filename)

    #Extract the data from fits file
    cont = hdu[0].data

    #Extract and correct header
    orig_header = hdu[0].header
    new_header = fits_header_clean(orig_header, grid_ra, grid_dec)

    #Start by building frequency and velocity arrays
    grid_wcs = WCS(new_header)
    
    if include_weights:
        wgts_filename = f'{dir_path}{grid_ra}+{grid_dec}{freq_slice}_continuumweights.fits'
        
        #Open the fits file
        wgts_hdu = fits.open(wgts_filename)
    
        #Extract the data from fits file
        wgts_cube = wgts_hdu[0].data

        #Extract and correct header
        orig_wgts_header = wgts_hdu[0].header
        new_wgts_header = fits_header_clean(orig_wgts_header, grid_ra, grid_dec)

        return cont, grid_wcs, new_header, wgts_cube, new_wgts_header
    else:
        return cont, grid_wcs, new_header