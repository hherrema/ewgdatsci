### Utility methods for lake data

# imports
import xarray as xr
import numpy as np


# ---------- Topography/Bathymetry ----------

def hypsographic_z(bathy, depth=None):
    """
    Create lake area depth profile.

    Parameters
    ----------
    bathy : xr.Dataset
        Lake bathymetry.
    depth : xr.DataArray
        Depth values from vertical profile.

    Returns
    -------
    area_z : xr.DataArray
        Lake area as function of depth [m^2].
    """
    if isinstance(depth, xr.DataArray):
        depth = depth.values
    else:
        depth = np.arange(bathy.depth.max())

    bathy_arr = bathy.depth.values.ravel()
    bathy_arr = bathy_arr[bathy_arr > 0]
    bathy_sorted = np.sort(bathy_arr)
    idx = np.searchsorted(bathy_sorted, depth, side='right')
    area = len(bathy_sorted) - idx

    # store in DataArray
    area_z = xr.DataArray(area, dims='depth', coords={'depth': depth}, name='area')

    return area_z


def center_of_volume(area_z):
    """
    Calculate center of volume of lake.
    
    Parameters
    ----------
    area_z : xr.DataArray
        Lake area as a function of depth [m^2].

    Returns
    -------
    zv : float
        Center of volume [m].
    """

    # center of volume
    zv = (area_z.depth * area_z).integrate(coord='depth') / area_z.integrate(coord='depth')
    
    return zv.item()


# ---------- Geometry ----------

def distance_btwn_points(x1, y1, x2, y2):
    """
    Calculate the distance between (x1, y1) and (x2, y2).

    Parameters
    ----------
    x1 : float
        X Swiss coordinate of point 1.
    y1 : float
        Y Swiss coordinate of point 1.
    x2 : float
        X Swiss coordinate of point 2.
    y2 : float
        Y Swiss coordinate of point 2.

    Returns
    -------
    distance : float
        Distance between coordinates [m].
    """
    x = x2 - x1
    y = y2 - y1

    return np.sqrt(x**2 + y**2)


# ---------- Trigonometry ----------

def angle_btwn_points(x1, y1, x2, y2, N_pos=True):
    """
    Calculate the angle a ray from (x1, y1) to (x2, y2) makes with the ray along the positive x-axis.

    Parameters
    ----------
    x1 : float
        X Swiss coordinate of point 1.
    y1 : float
        Y Swiss coordinate of point 1.
    x2 : float
        X Swiss coordinate of point 2.
    y2 : float
        Y Swiss coordinate of point 2.
    N_pos : bool
        If True, maintain northward as positive.

    Returns
    -------
    angle : float
        Angle transect between points makes with positive x-axis [°].
    """
    # maintain N-ward as positive
    if N_pos and y2 >= y1:
        x = x2 - x1
        y = y2 - y1
    else:
        x = x1 - x2
        y = y1 - y2

    return np.rad2deg(np.arctan2(y, x))


# ---------- Unit Conversions ----------

def ch1903_to_latlng(xsc, ysc):
    """
    Convert CH1903 Swiss coordinates to latitude and longitude.

    Parameters
    ----------
    xsc : int
        X Swiss coordinate.
    ysc : int
        Y Swiss coordinate.

    Returns
    -------
    lat : float
        Latitude [°].
    lon : float
        Longitude [°].
    """
    # remove leading 2 (xsc) and 1 (ysc)
    if xsc > 2e6:
        xsc = xsc - int(2e6)
    if ysc > 1e6:
        ysc = ysc - int(1e6)

    x = (xsc - 600000) / 1000000
    y = (ysc - 200000) / 1000000

    lat = (16.9023892 
        + 3.238272 * y 
        - 0.270978 * x ** 2 
        - 0.002528 * y ** 2 
        - 0.0447 * x ** 2 * y 
        - 0.014 * y ** 3)
    
    lng = (2.6779094 
        + 4.728982 * x 
        + 0.791484 * x * y 
        + 0.1306 * x * y ** 2 
        - 0.0436 * x ** 3)
    
    lat = (lat * 100) / 36
    lng = (lng * 100) / 36

    return lat, lng


def frequency_to_angular_frequency(f):
    """
    Convert frequency [1/s] to angular frequency [rad/s].
    
    Parameters
    ----------
    f : float
        Frequency [1/s].
    
    Returns
    -------
    w : float
        Angular frequency [rad/s].
    """
    return 2 * np.pi * f


def angular_frequency_to_frequency(w):
    """
    Convert angular frequency [rad/s] to frequency [1/s].

    Parameters
    ----------
    w : float
        Angular frequency [rad/s].

    Returns
    -------
    f : float
        Frequency [1/s].
    """
    return w / (2 * np.pi)


def s_to_hr(s):
    """
    Convert seconds to hours.

    Parameters
    ----------
    s : float
        Seconds [s].

    Returns
    -------
    hr : float
        Hours [hr].
    """
    return s / 3600


def hr_to_hz(period):
    """
    Convert hours (period) to Hz (frequency).

    Parameters
    ----------
    period : float
        Period [hr].

    Returns
    -------
    f : float
        Frequency [Hz].
    """
    return 1 / (period * 3600)


def hz_to_hr(f):
    """
    Convert Hz (frequency) to hours (period).

    Parameters
    ----------
    f : float
        Frequency [Hz].

    Returns
    -------
    period : float
        Period [hr].
    """
    return 1 / (f * 3600)