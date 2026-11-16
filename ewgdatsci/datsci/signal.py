### Methods for signal processing of lake data

# imports
import xarray as xr
import pandas as pd
import numpy as np
import scipy
import pywt


# ---------- Time Series ---------- #

def rolling_average_t():
    raise NotImplementedError


def rolling_integral_t(da, window_hr):
    """
    Compute rolling integral.

    Parameters
    ----------
    da : xr.DataArray
        Timeseries to run rolling integral over.
    window_hr : float
        Size of segments [hr].

    Returns
    -------
    da_ri : xr.DataArray
        Rolling integral.
    """
    # calculate sampling frequency, samples/hr
    speriod = da.time.diff(dim='time')[0].item() / 1e9
    sfreq_hr = int((1 / speriod) * 3600)

    # rolling windows
    roll = da.rolling(time=sfreq_hr * window_hr, center=False)
    window = roll.construct(time='t_window')

    return window.integrate('t_window')


def detrend(da, mode='linear', interp_na=True):
    """
    Detrend timeseries.

    Parameters
    ----------
    da : xr.DataArray
        Timeseries to detrend.
    mode : str
        Type of detrend.
    interp_na : bool
        If True, linear interpolation over NaN values (max 1 hour gap).

    Returns
    -------
    da_detrend : xr.DataArray
        Detrended timeseries.
    """
    if mode not in ['linear', 'constant']:
        raise ValueError('mode must be linear or constant.')

    # interpolate to fill NaN
    if interp_na:
        da = da.interpolate_na(dim='time', method='linear', max_gap='1hr')

    da_detrend = scipy.signal.detrend(da, type=mode)

    return xr.DataArray(da_detrend, dims=['time'], coords={'time': da.time}, name=da.name)


def detrend_z(da, mode='linear', interp_na=True):
    """
    Detrend timeseries with depth dimension. Detrend each depth independently.

    Parameters
    ----------
    da : xr.DataArray
        Timeseries with depth dimension to detrend.
    mode : str
        Type of detrend.
    interp_na : bool
        If True, linear interpolation over NaN values (max 1 hour gap).

    Returns
    -------
    da_detrend_z : xr.DataArray
        Detrended timeseries as function of depth.
    """
    da_detrend_z = []
    for d in da.depth.values:
        da_detrend = detrend(da.sel(depth=d), mode, interp_na).expand_dims({'depth': [d]})
        da_detrend_z.append(da_detrend)

    da_detrend_z = xr.concat(da_detrend_z, dim='depth')

    return da_detrend_z


def cross_correlation(da1, da2, mode='full', pandas=False):
    """
    Compute cross correlation between two timeseries.
    Positive lag indicates da1 follows da2.

    Parameters
    ----------
    da1 : xr.DataArray
        Timeseries data.
    da2 : xr.DataArray
        Timeseries data.
    mode : str
        Correlation method determining zero-padding.
    pandas : bool
        If True, convert to pandas dataframe.

    Returns:
    xcorr : xr.DataArray
        Cross correlation as function of temporal lag.
    """
    speriod = da1.time.diff(dim='time')[0].item() / 1e9

    corr = scipy.signal.correlate(da1, da2, mode=mode)
    lag_idx = scipy.signal.correlation_lags(len(da1), len(da2), mode=mode)
    lag = pd.to_timedelta(lag_idx * speriod, unit='s')
    lag_s = lag_idx * speriod
    lag_hr = lag_s / 3600

    xcorr = xr.DataArray(corr, dims=["lag"], coords={'lag': lag, 'lag_s': ('lag', lag_s), 'lag_hr': ('lag', lag_hr)}, name='cross_correlation')

    if pandas:
        xcorr = xcorr.to_dataframe().reset_index()

    return xcorr


# ---------- Frequency Filtering ---------- #

def lowpass_filter(da, f_thresh, order=4, interp_na=True, max_gap='1hr'):
    """
    Lowpass filter to remove frequencies above threshold.

    Parameters
    ----------
    da : xr.DataArray
        Signal to filter.
    f_thresh : float
        Threshold frequency [Hz].
    order : int
        Filter order.
    interp_na : bool
        If True, linear interpolation over NaN values.
    max_gap : str
        Maximum gap for linear interpolation of NaN values.

    Returns
    -------
    da_lp : xr.DataArray
        Lowpassed signal.
    """
    # calculate sampling frequency/period
    speriod = da.time.diff(dim='time')[0].item() / 1e9
    sfreq = 1 / speriod

    # interpolate to fill NaN
    if interp_na:
        da = da.interpolate_na(dim='time', method='linear', max_gap=max_gap)
    
    sos = scipy.signal.butter(order, f_thresh, btype='lowpass', analog=False, output='sos', fs=sfreq)

    da_lp = scipy.signal.sosfiltfilt(sos, da)

    return xr.DataArray(da_lp, dims=['time'], coords={'time': da.time}, name=da.name)


def lowpass_filter_z(da, f_thresh, order=4, interp_na=True):
    """
    Lowpass filter to remove frequencies above threshold across all depths.

    Parameters
    ----------
    da : xr.DataArray
        Signal to filter with depth dimension.
    f_thresh : float
        Threshold frequency [Hz].
    order : int
        Filter order.
    interp_na : bool
        If True, linear interpolation over NaN values (max 1 hour gap).
    
    Returns
    -------
    da_lp_z : xr.DataArray
        Lowpassed signal as function of depth.
    """
    da_lp_z = []
    for d in da.depth.values:
        da_lp = lowpass_filter(da.sel(depth=d), f_thresh, order, interp_na).expand_dims({'depth': [d]})
        da_lp_z.append(da_lp)

    da_lp_z = xr.concat(da_lp_z, dim='depth')

    return da_lp_z


def highpass_filter(da, f_thresh, order=4, interp_na=True):
    """
    Highpass filter to remove frequencies below threshold.

    Parameters
    ----------
    da : xr.DataArray
        Signal to filter.
    f_thresh : float
        Threshold frequency [Hz].
    order : int
        Filter order.
    interp_na : bool
        If True, linear interpolation over NaN values (max 1 hour gap).

    Returns
    -------
    da_hp : xr.DataArray
        Highpassed signal.
    """
    # calculate sampling frequency/period
    speriod = da.time.diff(dim='time')[0].item() / 1e9
    sfreq = 1 / speriod

    # interpolate to fill NaN
    if interp_na:
        da = da.interpolate_na(dim='time', method='linear', max_gap='1hr')
    
    sos = scipy.signal.butter(order, f_thresh, btype='highpass', analog=False, output='sos', fs=sfreq)

    da_hp = scipy.signal.sosfiltfilt(sos, da)

    return xr.DataArray(da_hp, dims=['time'], coords={'time': da.time}, name=da.name)


def bandpass_filter(da, f_band, order=4, interp_na=True):
    """
    Bandpass filter to select frequencies within range.

    Parameters
    ----------
    da : xr.DataArray
        Signal to filter.
    f_band : array_like
        Freqeuncy range to select [Hz].
    order : int
        Filter order.
    interp_na : bool
        If True, linear interpolation over NaN values (max 1 hour gap).

    Returns
    -------
    da_bp : xr.DataArray
        Bandpassed signal.
    """
    # calculate sampling frequency/period
    speriod = da.time.diff(dim='time')[0].item() / 1e9
    sfreq = 1 / speriod

    # interpolate to fill NaN
    if interp_na:
        da = da.interpolate_na(dim='time', method='linear', max_gap='1hr')
    
    sos = scipy.signal.butter(order, f_band, btype='bandpass', analog=False, output='sos', fs=sfreq)

    da_bp = scipy.signal.sosfiltfilt(sos, da)

    return xr.DataArray(da_bp, dims=['time'], coords={'time': da.time}, name=da.name)


# ---------- Spectral Analysis ---------- #

def psd_welch(da, window_hr):
    """
    Compute power spectral density using Welch's method.

    Parameters
    ----------
    da : xr.DataArray
        Time series data.
    window_hr : float
        Size of segments [hr].

    Returns
    -------
    psd : xr.DataArray
        Spectral power density.
    """
    # calculate sampling frequency/period
    speriod = da.time.diff(dim='time')[0].item() / 1e9
    sfreq = 1 / speriod
    nps = (3600 / speriod) * window_hr

    # interpolate to fill NaN
    da = da.interpolate_na(dim='time', method='linear', max_gap='1hr')

    # spectral decomposition
    freqs, psd = scipy.signal.welch(da.values, fs=sfreq, nperseg=nps, detrend='linear')
    
    freqs = freqs[1:]
    period = (1/freqs) / 3600
    psd = psd[1:]

    return xr.DataArray(psd, dims=['frequency'], coords={'frequency': freqs, 'period': (['frequency'], period)}, name="power_spectral_density")


def psd_welch_t_windows(da, window_hr, twfreq):
    """
    Compute spectral power using Welch's method.
    Manually define segments to average (minimizes disruption of NaN values).

    Parameters
    ----------
    da : xr.DataArray
        Time series data.
    window_hr : float
        Size of segments [hr].
    twfreq : str
        Frequency of time windows.

    Returns
    -------
    psd : xr.DataArray
        Spectral power density.
    """
    # time windows
    t_windows = pd.date_range(da.time.values[0], da.time.values[-1] - pd.Timedelta(hours=window_hr), freq=twfreq)

    psd_t = []
    for i, t in enumerate(t_windows):
        try:
            da_sel = da.sel(time=slice(t, t + pd.Timedelta(hours=window_hr, seconds=1)))[:-1]    # 1-second to handle clock drift?
            psd = psd_welch(da_sel, window_hr).expand_dims({'window': [i]})
            psd_t.append(psd)
        except ValueError:
            continue

    return xr.concat(psd_t, dim='window').mean(dim='window')


def psd_welch_z(da, window_hr, t_windows=False, twfreq=None, pandas=False, **kwargs):
    """
    Compute power spectral density using Welch's method across all depths.
    
    Parameters
    ----------
    da : xr.DataArray
        Time series data with depth dimension.
    window_hr : float
        Size of segments [hr].
    t_windows : bool
        If True, use psd_welch_t_windows method.
    twfreq : str
        Frequency of time windows.
    pandas : bool
        If True, convert to pandas dataframe.
    **kwargs : dict
        Metadata to assign as xarray attributes or pandas columns.

    Returns
    -------
    psd_z : xr.DataArray
        Spectral power density as function of depth.
    """
    psd_z = []
    for d in da.depth.values:
        try:
            if t_windows:
                psd = psd_welch_t_windows(da.sel(depth=d), window_hr, twfreq).expand_dims({'depth': [d]})
            else:
                psd = psd_welch(da.sel(depth=d), window_hr).expand_dims({'depth': [d]})

            psd_z.append(psd)
        except ValueError:
            continue

    psd_z = xr.concat(psd_z, dim='depth')

    if pandas:
        psd_z = psd_z.to_dataframe().reset_index()
        for k, v in kwargs.items():
            psd_z[k] = v
    else:
        psd_z = psd_z.assign_attrs(kwargs)

    return psd_z


def power_morlet_wavelets(da, freqs, interp_na=True, bw=1.5, cfreq=1.0, coi=False):
    """
    Compute spectral power via morlet wavelets.

    Parameters
    ----------
    da : xr.DataArray
        Time series for spectral analysis.
    freqs : np.array
        Target frequencies [Hz].
    interp_na : bool
        If True, linear interpolate over NaN values (max 1 hour gap).
    bw : float
        Wavelet bandwidth.
    cfreq : float
        Wavelet center frequency.
    coi : bool
        If True, mask out cone of influence.

    Returns
    -------
    pows : xr.DataArray
        Spectral power as function of time and frequency.
    """
    wvt = f'cmor{bw}-{cfreq}'

    speriod = da.time.diff(dim='time')[0].item() / 1e9
    period = (1 / freqs) / 3600 

    # interpolate to fill NaN
    if interp_na:
        da = da.interpolate_na(dim='time', method='linear', max_gap='1hr')

    scales = pywt.frequency2scale(wvt, freqs * speriod)    # normalized to sfreq = 1

    coeffs, _ = pywt.cwt(da, scales, wvt, sampling_period=speriod, method='fft')    # returns freqs
    pows = np.abs(coeffs)**2
    pows = xr.DataArray(pows, dims=['frequency', 'time'], coords={'time': da.time, 'frequency': freqs, 'period': (['frequency'], period)}, name='power')

    # cone of influence (mask boundary effects)    # may need wider mask
    if coi:
        eft = xr.DataArray(np.sqrt(bw) * scales * speriod , dims='frequency', coords={'frequency': freqs})    # e-folding time
        eft = xr.DataArray(eft, dims='frequency', coords={'frequency': freqs})
        time_s = (da.time - da.time[0]).astype('timedelta64[s]').astype(float)
        dist_to_bounds = np.minimum(time_s, time_s[-1] - time_s)
        coi = dist_to_bounds >= eft   # False = boundary effect
        pows = pows.where(coi)

    return pows

    
def power_morlet_wavelets_z(da, freqs, interp_na=True, bw=1.5, cfreq=1.0, coi=False):
    """
    Compute spectral power via morlet wavelets across all depths.

    Parameters
    ----------
    da : xr.DataArray
        Time series for spectral analysis with depth dimension.
    freqs : np.array
        Target frequencies [Hz].
    interp_na : bool
        If True, linear interpolate over NaN values (max 1 hour gap).
    bw : float
        Wavelet bandwidth.
    cfreq : float
        Wavelet center frequency.
    coi : float
        If True, mask out cone of influence.

    Returns
    -------
    pows_z : xr.DataArray
        Spectral power as function of time, frequency, and depth.
    """
    pows_z = []
    for d in da.depth.values:
        pows = power_morlet_wavelets(da.sel(depth=d), freqs, interp_na, bw, cfreq, coi).expand_dims({'depth': [d]})
        pows_z.append(pows)

    pows_z = xr.concat(pows_z, dim='depth')

    return pows_z


def log_z_power(pows):
    """
    Log-transform, then Z-score powers (within frequency and depth).
    For comparisons across frequencies with log-normal power distrbutions.
    
    Parameters
    ----------
    pows : xr.DataArray
        Spectral power as function of frequency (and depth).

    Returns
    -------
    pows_z : xr.DataArray
        Z-scored log powers.
    """
    pows_log = np.log10(pows)
    pows_z = (pows_log - pows_log.mean(dim='time')) / pows_log.std(dim='time', ddof=1)

    return pows_z


def rotary_psd(u, v):
    """
    Calculate rotary power spectral density.

    Parameters
    ----------
    u : xr.DataArray
        E-W velocity [m/s].
    v : xr.DataArray
        N-S velocity [m/s].

    Returns
    -------
    rpsd : xr.DataArray
        Rotary power spectral density.
    """
    # calculate sampling frequency/period
    speriod = u.time.diff(dim='time')[0].item() / 1e9
    sfreq = 1 / speriod

    # interpolate to fill NaN
    u = u.interpolate_na(dim='time', method='linear', max_gap='1hr')
    v = v.interpolate_na(dim='time', method='linear', max_gap='1hr')

    # complex velocity vector
    complex_vel = u + 1j * v
    
    # spectral decomposition
    freqs = scipy.fft.fftfreq(len(complex_vel), d=1/sfreq)
    fft = scipy.fft.fft(complex_vel)
    rpsd = (np.abs(fft) ** 2) / (len(complex_vel) * sfreq)

    freqs = freqs[1:]
    period = (1/freqs) / 3600
    rpsd = rpsd[1:]

    rpsd = xr.DataArray(rpsd, dims=['frequency'], coords={'frequency': freqs, 'period': (['frequency'], period)}, name='rotary_power_spectral_density')

    return rpsd.sortby('frequency')


def rotary_psd_t_windows(u, v, window_hr, twfreq):
    """
    Calculate rotary power spectral density.
    Manually define segments to average (minimizes disruption of NaN values).

    Parameters
    ----------
    u : xr.DataArray
        E-W velocity [m/s].
    v : xr.DataArray
        N-S velocity [m/s].
    window_hr : float
        Size of segments [hr].
    twfreq : str
        Frequency of time windows.

    Returns
    -------
    rpsd : xr.DataArray
        Rotary spectral power density.
    """
    # time windows
    t_windows = pd.date_range(u.time.values[0], u.time.values[-1] - pd.Timedelta(hours=window_hr), freq=twfreq)

    rpsd_t = []
    for i, t in enumerate(t_windows):
        try:
            u_sel = u.sel(time=slice(t, t + pd.Timedelta(hours=window_hr)))
            v_sel = v.sel(time=slice(t, t + pd.Timedelta(hours=window_hr)))
            rpsd = rotary_psd(u_sel, v_sel).expand_dims({'window': [i]})
            rpsd_t.append(rpsd)
        except BaseException as e:
            print(e)
            continue
    
    return xr.concat(rpsd_t, dim='window').mean(dim='window')


def rotary_psd_z(u, v, t_windows=False, window_hr=None, twfreq=None, pandas=False, **kwargs):
    """
    Calculate rotary power spectral density across all depths.
    
    Parameters
    ----------
    u : xr.DataArray
        E-W velocity with depth dimension [m/s].
    v : xr.DataArray
        N-S velocity with depth dimension [m/s].
    t_windows : bool
        If True, use psd_welch_t_windows method.
    window_hr : float
        Size of segments [hr].
    twfreq : str
        Frequency of time windows.
    pandas : bool
        If True, convert to pandas dataframe.
    **kwargs : dict
        Metadata to assign as xarray attributes or pandas columns.

    Returns
    -------
    rpsd_z : xr.DataArray
        Rotary spectra power density as function of depth.
    """
    rpsd_z = []
    for d in u.depth.values:
        if t_windows:
            rpsd = rotary_psd_t_windows(u.sel(depth=d), v.sel(depth=d), window_hr, twfreq).expand_dims({'depth': [d]})
        else:
            rpsd = rotary_psd(u.sel(depth=d), v.sel(depth=d)).expand_dims({'depth': [d]})
        
        rpsd_z.append(rpsd)

    rpsd_z = xr.concat(rpsd_z, dim='depth')
    rpsd_z = rpsd_z.dropna(dim='depth', how='all')

    if pandas:
        rpsd_z = rpsd_z.to_dataframe().reset_index()
        
        # anticyclonic v. cyclonic
        rpsd_z['direction'] = np.where(rpsd_z['frequency'] > 0, 'CCW', 'CW')
        rpsd_z[['abs_frequency', 'abs_period']] = abs(rpsd_z[['frequency', 'period']])

        for k, v in kwargs.items():
            rpsd_z[k] = v
    else:
        rpsd_z = rpsd_z.assign_attrs(kwargs)

    return rpsd_z


def rotary_power_morlet_wavelets(u, v, freqs, interp_na=True, bw=1.5, cfreq=1.0, coi=False):
    """
    Calculate rotary spectral power via morlet wavelets.

    Parameters
    ----------
    u : xr.DataArray
        E-W velocity [m/s].
    v : xr.DataArray
        N-S velocity [m/s].
    freqs : np.array
        Target frequencies [Hz].
    interp_na : bool
        If True, linear interpolate over NaN values (max 1 hour gap).
    bw : float
        Wavelet bandwidth.
    cfreq : float
        Wavelet center frequency.
    coi : float
        If True, mask out cone of influence.

    Returns
    -------
    pows_ccw : xr.DataArray
        Counterclockwise rotary spectral power as function of time and frequency.
    pows_cw : xr.DataArray
        Clockwise rotary spectral power as function of time and frequency.
    """
    # interpolate to fill NaN, only do on real components
    if interp_na:
        u = u.interpolate_na(dim='time', method='linear', max_gap='1hr')
        v = v.interpolate_na(dim='time', method='linear', max_gap='1hr')

    # complex -> CCW
    complex_vel = u + 1j * v
    pows_ccw = power_morlet_wavelets(complex_vel, freqs, False, bw, cfreq, coi)

    # complex conjugate -> CW
    conjugate_vel = u - 1j * v
    pows_cw = power_morlet_wavelets(conjugate_vel, freqs, False, bw, cfreq, coi)

    return pows_ccw, pows_cw


def rotary_power_morlet_wavelets_z(u, v, freqs, interp_na=True, bw=1.5, cfreq=1.0, coi=False):
    """
    Calculate rotary spectral power via morlet wavelets across all depths.

    Parameters
    ----------
    u : xr.DataArray
        E-W velocity as function of depth [m/s].
    v : xr.DataArray
        N-S velocity as function of depth [m/s].
    freqs : np.array
        Target frequencies [Hz].
    interp_na : bool
        If True, linear interpolate of NaN values (max 1 hour gap).
    bw : float
        Wavelet bandwidth.
    cfreq : float
        Wavelet center frequency.
    coi : float
        If True, mask out cone of influence.

    Returns
    -------
    pows_ccw_z : xr.DataArray
        Counterclockwise rotary spectral power as function of time, frequency, and depth.
    pows_cw_z : xr.DataArray
        Clockwise rotary spectral power as function of time, frequency, and depth.
    """
    pows_ccw_z, pows_cw_z = [], []
    for d in u.depth.values:
        pows_ccw, pows_cw = rotary_power_morlet_wavelets(u.sel(depth=d), v.sel(depth=d), freqs, interp_na, bw, cfreq, coi)
        pows_ccw_z.append(pows_ccw.expand_dims({'depth': [d]}))
        pows_cw_z.append(pows_cw.expand_dims({'depth': [d]}))

    pows_ccw_z = xr.concat(pows_ccw_z, dim='depth')
    pows_cw_z = xr.concat(pows_cw_z, dim='depth')

    return pows_ccw_z, pows_cw_z


# ---------- Depth Profiles ---------- #

def rolling_average_z(da, stat, ra_window=1):
    """
    Compute rolling mean or median along depth dimension.

    Parameters
    ----------
    da : xr.DataArray
        Data to compute rolling average of.
    stat : str
        Statistic to compute for each rolling segment.
    ra_window : float
        Depth window for rolling average [m].

    Returns
    -------
    da_ra : xr.DataArray
        Rolling average of data.
    """
    if stat == 'mean':
        da_ra = [da.sel(depth=slice(d - (ra_window/2), d + (ra_window/2))).mean() for d in da.depth]
    elif stat == 'median':
        da_ra = [da.sel(depth=slice(d - (ra_window/2), d + (ra_window/2))).median() for d in da.depth]
    else:
        raise ValueError('Only rolling mean and median implemented.')
    
    return xr.DataArray(da_ra, dims=da.dims, coords=da.coords, name=da.name)


def binned_average_z(da, stat, bin_size=1):
    """
    Compute binned average along depth dimension.

    Parameters
    ----------
    da : xr.DataArray
        Data to compute binned average of.
    stat : str
        Statistic to bompute for each bin.
    bin_size : float
        Depth bin size [m].

    Returns
    -------
    da_ba : xr.DataArray
        Binned average of data.
    """
    bins = np.arange(0, da.depth.max() + bin_size, bin_size)

    if stat == 'mean':
        da_ba = da.groupby_bins('depth', bins, labels=bins[:-1] + bin_size/2).mean()
    elif stat == 'median':
        da_ba = da.groupby_bins('depth', bins, labels=bins[:-1] + bin_size/2).median()
    else:
        raise ValueError('Only binned mean and median implemented.')
    
    return da_ba.rename({'depth_bins': 'depth'})


def order_profile(da, surfmax, reassign_depth=False):
    """
    Order profile.  Stable so repeat values maintain original order.

    Parameters
    ----------
    da : xr.DataArray
        Profile to sort.
    surfmax : bool
        True if var is max at surface, False if var is max at bottom.
    reassign_depth: bool
        If True, reassign original depth dimension.

    Returns
    -------
    da_sorted : xr.DataArray
        Ordered profile.
    """
    if surfmax:
        s = -1
    else:
        s = 1

    # extract depth before sorting
    depth = da['depth']

    da_ascending = da * s
    idx = da_ascending.argsort(kind='mergesort').values
    da_sorted = da_ascending.isel(depth=idx) * s

    if reassign_depth:
        da_sorted = da_sorted.assign_coords(depth=depth)

    return da_sorted


# ---------- Depth-Binned Data ---------- #

def valid_depths(ds, thresh):
    """
    Filter depths below threshold of non-nan values.
    Currently implemented for ADCP data.

    Parameters
    ----------
    ds : xr.Dataset
        Data with a depth dimension.
    thresh : float
        Theshold for percentage of data with non-nan values.

    Returns
    -------
    ds : xr.Dataset
        Filtered data with only valid depths.
    """
    prop_valid = ds['vel'].notnull().mean(dim=['time' ,'dir'])

    return ds.sel(depth=prop_valid >= thresh)


# ---------- Triginometry ---------- #

def velocity_projection(vel, angle):
    """
    Project ADCP velocity along angle, where angle is defined with respect to 0° = E.

    Parameters
    ----------
    vel : xr.DataArray
        Velocity timeseries [m/s].
    angle: float
        Angle from x-axis [°].

    Returns
    -------
    vel_proj : xr.DataArray
        Velocity projection.
    """
    return vel.sel(dir='E') * np.cos(np.deg2rad(angle)) + vel.sel(dir='N') * np.sin(np.deg2rad(angle))


def wind_projection(wind_speed, wind_direction, angle):
    """
    Project wind along angle, where angle is defined with respect to 0° = E.
    wind_direction argument must follow MeteoSwiss convention: 0° = Northerly, 90° = Easterly.
    angle = 90° will return wind towards North (Southerly).

    Parameters
    ----------
    wind_speed : xr.DataArray
        Wind speed timeseries [m/s].
    wind_direction : xr.DataArray
        Wind direction timeseries [°].
    angle: float
        Angle from x-axis [°].

    Returns
    -------
    wind_proj : xr.DataArray
        Wind projection [m/s].
    """
    wind_dir_rad = np.deg2rad(wind_direction)
    u_wind = -wind_speed * np.sin(wind_dir_rad)
    v_wind = -wind_speed * np.cos(wind_dir_rad)
    wind_proj = u_wind * np.cos(np.deg2rad(angle)) + v_wind * np.sin(np.deg2rad(angle))

    return wind_proj.rename('wind')



# ---------- Misc ---------- #

def savitzky_golay(arr):
    """
    Apply Savitzky-Golay filter to smooth array.

    Parameters
    ----------
    arr : array_like
        Array of values to smooth.

    Returns
    -------
    arr_sg : array_like
        Smoothed array.
    """
    window = int(np.ceil(len(arr)/10) // 2 * 2 + 1)
    polyorder = min(3, window)
    
    return scipy.signal.savgol_filter(arr, window, polyorder, mode='nearest')





