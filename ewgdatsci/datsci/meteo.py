### Meteorologica data

# imports
import numpy as np
import pandas as pd


def locate_wind_evs_contiguous(wind_speed, thresh):
    """
    Locate contiguous periods with wind speeds above a threshold.

    Parameters
    ----------
    wind_speed : xr.DataArray
        Wind speed [m/s].
    thresh : float
        Threshold for high winds [m/s].

    Returns
    -------
    wind_evs : pd.DataFrame
        Contiguous periods longer than 1 hour of high winds.
    """
    mask = wind_speed > thresh

    # locate start and end times of high winds
    diff = mask.astype(int).diff(dim='time', label='upper')
    t_start = diff.time.where(diff == 1, drop=True).values
    t_end = diff.time.where(diff == -1, drop=True).values

    # edge cases, wind events at start/end of timeseries
    if mask.isel(time=0) and (len(t_start) == 0 or t_start[0] > mask.time[0]):
        t_start = np.insert(t_start, 0, mask.time[0].values)
    if mask.isel(time=-1) and (len(t_end) == 0 or t_end[-1] < mask.time[-1]):
        t_end = np.append(t_end, mask.time[-1].values)

    # filter by duration longer than 1 hour
    wind_evs = pd.DataFrame({'thresh': thresh, 't_start': t_start, 't_end': t_end})
    wind_evs['duration'] = wind_evs['t_end'] - wind_evs['t_start']
    wind_evs = wind_evs[wind_evs['duration'] > '01:00:00']

    # average wind speed during wind event
    wind_evs["wind_speed"] = wind_evs.apply(lambda x: wind_speed.sel(time=slice(x.t_start, x.t_end)).mean().item(), axis=1)

    return wind_evs.sort_values('duration', ascending=False).reset_index(drop=True)


def locate_wind_evs_window(wind_speed, window_hr, thresh):
    """
    Locate periods of a given duration with average wind speeds above a threshold.

    Parameters
    ----------
    wind_speed : xr.DataArray
        Wind speed [m/s].
    window_hr : int
        Window for rolling average [hr].
    thresh : float
        Thresold for high winds [m/s].

    Returns
    -------
    wind_evs : pd.DataFrame
        Periods with high winds.
    """
    lbl = wind_speed.name
    wind_speed_ra = wind_speed.rolling(time=6*window_hr, center=True).mean()
    wind_evs = wind_speed_ra.where(wind_speed_ra > thresh, drop=True).to_dataframe().reset_index()

    wind_evs['t_start'] = wind_evs['time'] - pd.Timedelta(hours=window_hr/2)
    wind_evs['t_end'] = wind_evs['time'] + pd.Timedelta(hours=window_hr/2)
    wind_evs[['window_hr', 'thresh']] = window_hr, thresh

    return wind_evs.sort_values(lbl, ascending=False).reset_index(drop=True)


def wind_energy_flux(W10, Cd=0.9e-3):
    """
    Calculate downwards wind energy flux at 10 meters above surface.
    P10 = rho_air * Cd * W10^3

    Parameters
    ----------
    W10 : xr.DataArray
        Wind speed 10 m above lake surface [m/s].
    Cd : float
        Drag coefficient.

    Returns
    -------
    P10 : xr.DataArray
        Wind energy flux [mW/m^2].
    """
    rho_air = 1.2    # air density
    P10 = rho_air * Cd * (W10**3)

    return P10.rename('P10')
    