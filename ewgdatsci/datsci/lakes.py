### Methods for analyzing lake data

# imports
import xarray as xr
import numpy as np
import pandas as pd
import scipy
import math

from .signal import rolling_average_z, order_profile


# ---------- Stratification ---------- #

def mixed_layer_depth(rho, thresh=0.1):
    """
    Locate depth of mixed layer from surface.
    Defined as depth at which density exceeds surface density by given threshold.
    Density profile should be smoothed.

    Parameters
    ----------
    rho : xr.DataArray
        Density profile.
    thresh : float
        Density difference from surface density [kg/m^3].

    Returns
    -------
    mld : float
        Mixed layer depth.
    """
    # surface water density
    rho_surf = rho.isel(depth=0).item()

    # mixed layer depth
    mask = rho > rho_surf + thresh
    mld = rho['depth'].where(mask).min().item()
    
    return mld


def schmidt_stability(ds, bathy):
    """
    Calculate Schmidt stability.
    St = (g / A0) * int{0^zm}(z - zv) * rhoz * Az dz

    Parameters
    ----------
    ds : xr.Dataset
        CTD data.
    bathy : xr.Dataset
        Lake Bathymetry

    Returns
    -------
    St : float
        Schmidt stability.
    """
    g = 9.81
    # surface area
    A0 = (bathy.depth > 0).sum().item()

    mask = (ds['depth'].notnull()) & (ds['Temp_qual'] == 0) & (ds['Cond_qual'] == 0)
    depth = ds['depth'][mask]
    rho = ds['rho'][mask]

    # calculate center of volume and area at each depth
    zv, area_z = center_of_volume(bathy, depth)
    
    integral = xr.DataArray(((depth.values - zv) * rho.values * area_z), dims='depth', coords={'depth': depth.values}, name='St_integrand').integrate(coord='depth')
    St = (g / A0) * integral

    return St.item()


def thermocline(ds, ra_window=1):
    """
    Locate thermocline, defined as maximum vertical temperature gradient.

    Parameters
    ----------
    ds : xr.Dataset 
        Single profile CTD data.
    ra_window : float
        Depth window for rolling average.

    Returns
    -------
    thermocline_depth : float
        Depth of thermocline.
    """
    # only keep samples with valid depths, quality checked temperature
    mask = ds['depth'].notnull() & (ds['Temp_qual'] == 0)
    depth = ds['depth'][mask].values
    temp = ds['Temp'][mask].values

    # rolling average to smooth temperature profile
    temp_ra = rolling_average(temp, depth, ra_window)

    # locate maximum negative temperature gradient
    dTdz = np.gradient(temp_ra, depth)
    thermocline_idx = np.nanargmin(dTdz)
    thermocline_depth = depth[thermocline_idx]

    return thermocline_depth


def epi_meta_hypolimnion(ds):
    """
    Locate epilimnion, metalimnion, and hypolimnion regions, based on temperature profile.

    Parameters
    ----------
    ds : xr.Dataset
        Single profile CTD data.

    Returns
    -------
    epilimnion : tuple
        Start and end depths of epilimnion.
    metalimnion : tuple
        Start and end depths of metalimnion.
    hypolimnion : tuple
        Start and end depths of hypolimnion.
    """
    raise NotImplementedError


# ---------- Isolines ---------- #

def isoline(da, iso_val):
    """
    Locate iso-(therm/pycnal/oxygen) depths.  
    Calculate as crossing of iso_val.

    Parameters
    ----------
    da : xr.DataArray
        Profile from CTD.
    iso_val : float
        Isoline value.

    Returns
    -------
    iso_z : np.array
        Isoline depths [m].
    """
    depth = da['depth']

    # baseline correct by iso_val
    da_bc = da - iso_val

    # find 0 crossing (adjacent measurements with opposite sign)
    mask = (da_bc * da_bc.shift(depth=1, fill_value=0)) < 0

    return depth.where(mask, drop=True).values


def isoline_t(da, iso_val):
    """
    Locate isoline from timeseries.  Calculate as max depth of crossings of iso_val.
    TO DO: Offer mean depth return.
    
    Parameters
    ----------
    da : xr.DataArray
        Timeseries data from vertical sensor chain.
    iso_val : float
        Isoline value.

    Returns
    -------
    iso_z : xr.DataArray
        Isoline depth timeseries [m].
    """
    # interpolate to 1 meter depth bins
    interp_z = np.arange(np.floor(da.depth.min()), np.ceil(da.depth.max()) + 1, 1)
    interp_da = da.interp(depth=interp_z, method='linear')
    depth = interp_da['depth']

    # baseline correct by iso_val
    da_bc = interp_da - iso_val

    # find 0 crossing (adjacent measurements with opposite sign)
    mask = (da_bc * da_bc.shift(depth=1, fill_value=0)) < 0

    return depth.where(mask).max(dim='depth').set_index(time='time')


# ---------- Anoxia ---------- #

def transect_min_dox(datasets):
    """
    Calculate minimum dissolved oxygen concentration across profiles from same transect.

    Parameters
    ----------
    datasets : list
        Datasets for profiles taken during same transect.

    Returns
    -------
    min_dox : float
        Minimum dissolved oxygen concentration during transect.
    min_dox_depth: float
        Depth of minimum dissolved oxygen concentraton.
    max_transect_depth : float
        Maximum depth during transect.
    """
    min_dox, min_dox_depth, max_transect_depth = np.inf, np.inf, -np.inf
    for ds in datasets:
        # only keep samples with valid depths, quality checked dissolved oxygen
        mask = ds['depth'].notnull() & (ds['DO_mg_qual'] == 0)
        depth = ds['depth'][mask]
        dox_conc = ds['DO_mg'][mask]

        # calculate profile values
        min_dox_p = dox_conc.min().item()
        min_dox_idx = np.where(dox_conc == min_dox_p)[0]
        min_dox_depth_p = depth[min_dox_idx].min().item()
        max_depth_p = depth.max().item()

        # update transect minimum dissolved oxygen concentration
        if min_dox_p < min_dox:
            min_dox = min_dox_p
            min_dox_depth = min_dox_depth_p

        # update transect maximum depth
        if max_depth_p > max_transect_depth:
            max_transect_depth = max_depth_p

    return min_dox, min_dox_depth, max_transect_depth


def anoxia(ds, epsilon=0.05, transect_min_dox=None):
    """
    Locate start of anoxic zone.

    Parameters
    ----------
    ds : xr.Dataset
        Single profile CTD data.
    epsilon : float
        Allowed deviation from minimum dissolved oxygen concentration.  SHOULD BE TAKEN AS MEASUREMENT ERROR OF CTD.
    transect_min_dox : float
        Minimum dissolved oxygen during transect.

    Returns
    -------
    anoxic_depth : float
        Shallowest depth of anoxic conditions.
    anoxic_dox : float
        Minimum dissolved oxygen concentration.
    anoxic_sem : float
        Standard error of dissolved oxygen measurements below anoxic depth.
    """
    # only keep samples with valid depths, quality checked dissolved oxygen
    mask = ds['depth'].notnull() & (ds['DO_mg_qual'] == 0)
    depth = ds['depth'][mask]
    dox_conc = ds['DO_mg'][mask]

    # compare with transect minimum
    if transect_min_dox:
        anoxic_idx = np.where(dox_conc <= transect_min_dox + epsilon)[0]
    # compare within profile
    else:        
        anoxic_idx = np.where(dox_conc <= min(dox_conc.min().item(), epsilon))[0]

    if len(anoxic_idx) == 0:
        raise IndexError('No anoxic depths located.')
    
    anoxic_depth = depth[anoxic_idx].min().item()
    anoxic_dox = dox_conc[anoxic_idx.min():]

    return anoxic_depth, np.min(anoxic_dox), scipy.stats.sem(anoxic_dox)


# ---------- Turbulence/Mixing ---------- #

def thorpe_displacement(depth, depth_sort):
    """
    Calculate Thorpe displacement.
    
    Parameters
    ----------
    depth : xr.DataArray
        Observed depth profile.
    depth_sort : np.array
        Depth profile from ordered variable.

    Returns
    -------
    dT : xr.DataArray
        Thorpe displacement [m].
    """
    dT = depth - depth_sort
    
    return dT.rename('thorpe_displacement')


def thorpe_displacement_rho(rho, depth, ra_window=1):
    """
    Calculate Thorpe displacement from density profile.

    Parameters
    ----------
    rho : xr.DataArray
        Density profile.
    depth : xr.DataArray
        Depth profile.
    ra_window : float
        Depth window for rolling average [m].

    Returns
    -------
    dT : xr.DataArray
        Thorpe displacement [m].
    """
    rho_ra = rolling_average_z(rho, 'mean', ra_window)
    rho_ra_sort = order_profile(rho_ra, surfmax=False, reassign_depth=False)

    dT = thorpe_displacement(depth, rho_ra_sort.depth.values)

    return dT


def thorpe_displacement_temp(temp, depth, ra_window=1):
    """
    Calculate Thorpe displacement from temperature profile.

    Parameters
    ----------
    temp : xr.DataArray
        Temperature profile.
    depth : xr.DataArray
        Depth profile.
    ra_window : float
        Depth window for rolling average [m].

    Returns
    -------
    dT : xr.DataArray
        Thorpe displacement [m].
    """
    temp_ra = rolling_average_z(temp, 'mean', ra_window)
    temp_ra_sort = order_profile(temp_ra, surfmax=True, reassign_depth=False)

    dT = thorpe_displacement(depth, temp_ra_sort.depth.values)

    return dT


def thorpe_scale(dT):
    """
    Calculate Thorpe scale as RMS of Thorpe displacement.

    Parameters
    ----------
    dT : xr.DataArray
        Thorpe displacement [m].

    Returns
    -------
    LT : float
        Thorpe scale [m].
    """
    return np.sqrt((dT**2).mean()).item()



# ---------- Volume Flux/Transport ---------- #

def locate_interface_bounds(depth, mxsc, mysc, angle_rad, bathy, oom=True):
    """
    Locate land boundaries of flux interface.

    TO DO: edge case where d < depth is not true land boundary (i.e., ridge).

    Parameters
    ----------
    depth : float
        Depth of ADCP bin.
    mxsc : int
        x Swiss coordinate of mooring location.
    mysc : int
        y Swiss coordinate of mooring location.
    angle_rad : float
        Angle of interface in radians.
    bathy : xr.Dataset
        Lake bathymetry.
    oom : bool
        Toggle to add order of magnitude (2, 1) to (x, y) coordinates.

    Returns
    -------
    xsc1 : float
        x Swiss coordinate of boundary 1.
    ysc1 : float
        y Swiss coordinate of boundary 1.
    xsc2 : float
        x Swiss coordinate of boundary 2.
    ysc2 : float
        y Swiss coordinate of boundary 2.
    distance : float
        Distance between boundary points.

    """
    if oom:
        mxsc = int(mxsc + 2e6)
        mysc = int(mysc + 1e6)

    # positive direction
    xsc1, ysc1 = mxsc, mysc
    while xsc1 < bathy.xsc.max().item() and ysc1 < bathy.ysc.max().item():
        d = bathy.sel(xsc=xsc1, ysc=ysc1, method='nearest').depth.item()

        # boundary reached
        if d < depth or math.isnan(d):
            break

        xsc1 = xsc1 + np.cos(angle_rad)
        ysc1 = ysc1 + np.sin(angle_rad)

    # negative direction
    xsc2, ysc2 = mxsc, mysc
    while xsc2 > bathy.xsc.min().item() and ysc2 > bathy.ysc.min().item():
        d = bathy.sel(xsc=xsc2, ysc=ysc2, method='nearest').depth.item()

        # boundary reached
        if d < depth or math.isnan(d):
            break

        xsc2 = xsc2 - np.cos(angle_rad)
        ysc2 = ysc2 - np.sin(angle_rad)

    # distance between boundary points
    distance = np.sqrt((xsc1 - xsc2)**2 + (ysc1 - ysc2)**2)

    return xsc1, ysc1, xsc2, ysc2, distance


def volume_flux(vel, depth, h, mxsc, mysc, angle_deg, bathy):
    """
    Calculate volumne flux from ADCP velocity data.

    Parameters
    ----------
    vel : xr.DataArray
        ADCP current velocity data [m/s].
    depth : float
        Depth of ADCP bin [m].
    h : float
        Depth bin cell size [m].
    mxsc : int
        X Swiss coordinate of mooring location.
    mysc : int
        Y Swiss coordinate of mooring location.
    angle_deg : float
        Angle of interface in degrees.  0° indicates E-W interface.  90° indicates N-S interface.
    bathy : xr.Dataset
        Lake bathymetry.

    Returns
    -------
    flux : xr.DataArray
        Flux timeseries across interface [m^3/s].
    """
    if angle_deg < -90 or angle_deg >= 90:
        raise ValueError("Interface angle must be in range [-90°, 90°)")
    
    angle_rad = np.deg2rad(angle_deg)
    xsc1, ysc1, xsc2, ysc2, distance = locate_interface_bounds(depth, mxsc, mysc, angle_rad, bathy)
    da = distance * h               # area of cross-section rectangle [m^2]

    u = vel.sel(dir='E')
    v = vel.sel(dir='N')

    u_orth = u * (-1) * np.sin(angle_rad)
    v_orth = v * np.cos(angle_rad)

    # flux = velocity * cross-section area
    flux = (u_orth + v_orth) * da
    
    return flux.rename('flux')


def volume_flux_z(vel, h, mxsc, mysc, angle_deg, bathy):
    """
    Calculate volume flux from ADCP velocity data across depths.

    Parameters
    ----------
    vel : xr.DataArray
        ADCP current velocity data [m/s].
    h : float
        Depth bin cell size [m].
    mxsc : int
        X Swiss coordinate of mooring location.
    mysc : int
        Y Swiss Coordinate of mooring location.
    angle_deg : float
        Angle of interface in degrees.  0° indicates E-W interface.  90° indicates N-S interface.
    bathy : xr.Dataset
        Lake bathymetry.

    Returns
    -------
    flux_z: xr.DataArray
        Flux timeseries across interface as function of depth [m^3/s].
    """
    flux_z = []
    for d in vel.depth.values:
        flux = volume_flux(vel.sel(depth=d), d, h, mxsc, mysc, angle_deg, bathy).expand_dims({'depth': [d]})
        flux_z.append(flux)

    flux_z = xr.concat(flux_z, dim='depth')

    return flux_z


def net_volume_transport(flux):
    """
    Calculate net volume transport as integral of volume flux timeseries.

    Parameters
    ----------
    flux : xr.DataArray
        Volume flux timeseries [m^3/s].

    Returns
    -------
    transport : xr.DataArray
        Net volume transport [m^3].
    """
    time_s = (flux.time - flux.time[0]).astype('timedelta64[s]').astype(float)
    transport = flux.assign_coords(time=time_s).fillna(0).integrate('time')

    # new implementation
    #transport = ds_flux.fillna(0).integrate('time', datetime_unit='s')

    return transport.rename('transport')


def cumulative_volume_transport(flux):
    """
    Calculate cumulative volume transport as running integral of volume flux timeseries.

    Parameters
    ----------
    flux : xr.DataArray
        Volume flux timeseries [m^3/s].
    
    Returns
    -------
    cvt : xr.DataArray
        Cumulative volume transport [m^3].
    """
    cvt = flux.fillna(0).cumulative_integrate(coord='time', datetime_unit='s')

    return cvt.rename('cumulative_transport')


def contiguous_unidirectional_flow(ds, depth, angle_deg):
    """
    Determine start and end of contiguous flows in the same direction.

    Parameters
    ----------
    ds : xr.Dataset
        ADCP data.
    depth : float
        Depth of ADCP bin.
    angle_deg : float
        Angle of interface in degrees.  0° indicates E-W interface.  90° indicates N-S interface.

    Returns
    -------
    cuf : pd.DataFrame
        Direction, start and end indices/times, and duration of each contiguous unidirectional flow.
    vel : xr.DataArray
        Velocity timeseries at specified depth and angle.
    """
    if angle_deg < -90 or angle_deg >= 90:
        raise ValueError("Interface angle must be in range [-90°, 90°)")
    
    angle_rad = np.deg2rad(angle_deg)
    u = ds.vel.sel(dir='E').sel(range=depth, method='nearest')
    v = ds.vel.sel(dir='N').sel(range=depth, method='nearest')
    u_orth = u * (-1) * np.sin(angle_rad)
    v_orth = v * np.cos(angle_rad)
    vel = u_orth + v_orth
    depth = vel['range'].item()

    i = 0
    cuf = []
    while i < len(vel):
        # nortward flow
        if vel[i].item() > 0:
            j = i

            # contiguous N-ward flow
            while j < len(vel) and vel[j].item() > 0:
                j += 1

            # only flows greater than a single measurement
            if j > i+1:
                v_sel = vel.isel(time=slice(i, j))
                time_s = (v_sel.time - v_sel.time[0]).astype('timedelta64[s]').astype(float)
                duration = time_s[-1].item() - time_s[0].item()
                cuf.append({
                    'direction': 'N',
                    'depth': depth,
                    'angle_deg': angle_deg,
                    'idx_start': i,
                    'idx_end': j,
                    't_start': v_sel.time.values[0],
                    't_end': v_sel.time.values[-1],
                    'duration': duration
                })

            i = j

        # southward flow
        elif vel[i].item() < 0:
            j = i

            # contiguous S-ward flow
            while j < len(vel) and vel[j].item() < 0:
                j += 1

            # only flows greater than a single measurement
            if j > i+1:
                v_sel = vel.isel(time=slice(i, j))
                time_s = (v_sel.time - v_sel.time[0]).astype('timedelta64[s]').astype(float)
                duration = time_s[-1].item() - time_s[0].item()
                cuf.append({
                    'direction': 'S',
                    'depth': depth,
                    'angle_deg': angle_deg,
                    'idx_start': i,
                    'idx_end': j,
                    't_start': v_sel.time.values[0],
                    't_end': v_sel.time.values[-1],
                    'duration': duration
                })

            i = j

        else:
            i += 1
    
    return pd.DataFrame(cuf), vel


def excursion_length(ds, depth, angle_deg):
    """
    Calculate distance a water particle moves during a contiguous period of unidirectional motion.
    
    Reference: LawrEtal1997

    Parameters
    ----------
    ds : xr.Dataset
        ADCP data.
    depth : float
        Depth of ADCP bin.
    angle_deg : float
        Angle of interface in degrees.  0° indicates E-W interface.  90° indicates N-S interface.

    Returns
    -------
    cuf : pd.DataFrame
        Contiguous unidirectional flow data with excursion lengths [m].
    """
    cuf, vel = contiguous_unidirectional_flow(ds, depth, angle_deg)

    el_vals = []
    for _, row in cuf.iterrows():
            v_sel = vel.isel(time=slice(row.idx_start, row.idx_end))
            time_s = (v_sel.time - v_sel.time[0]).astype('timedelta64[s]').astype(float)
            el = v_sel.assign_coords(time=time_s).integrate('time').item()
            el_vals.append(abs(el))
    cuf['excursion_length'] = el_vals

    return cuf


def flow_angle(vel):
    """
    Calculate angle of flow from ADCP velocity data.
    Map {'E', 'N', 'W', 'S'} to angles {0°, 90°, 180°, 270°}.

    Parameters
    ----------
    vel : xr.DataArray
        ADCP current velocity data.

    Returns
    -------
    flow_ang : xr.DataArray
        Flow angle timeseries [°].
    """
    u = vel.sel(dir='E')
    v = vel.sel(dir='N')

    flow_ang = np.rad2deg(np.arctan2(v, u)) % 360

    return flow_ang.rename('angle')


# ---------- Modal Analysis ---------- #

def vertical_modes(N2, nmodes, method='klinck'):
    """
    Calculate normal baroclinic modes based on statification.

    Parameters
    ----------
    N2 : xr.DataArray
        Buoyancy frequency depth profile.
    nmodes : int
        Number of vertical modes to calculate.
    method : string
        Calculation method.

    Returns
    -------
    ev : xr.DataArray
        Eigenvalues.
    u : xr.DataArray
        Normalized horizontal velocity as function of depth and mode.
    w : xr.DataArray
        Normalized vertical velocity as function of depth and mode.
    """
    if method not in ['klinck', 'kundu']:
        raise ValueError('method must be either klink or kundu.')

    g = 9.81
    
    # make z negative, increasing, surface boundary condition
    depth = N2.depth.values
    z = np.append(-depth[::-1], 0)
    N2 = np.append(N2.values[::-1], N2.values[0])

    A = np.zeros((len(z), len(z)))

    dzd = (z[2:] - z[:-2]) / 2
    dzu = z[2:] - z[1:-1]
    dzl = z[1:-1] - z[:-2]

    # for vectorization
    i = np.arange(1, len(z)-1)

    if method == 'klinck':
        # fill in matrices
        A[i, i+1] = (1 / dzd) * (1 / dzu)
        A[i, i] = -(1 / dzd) * (1 / dzl + 1 / dzu)
        A[i, i-1] = (1 / dzd) * (1 / dzl)

        B = np.diag(-N2)

        # boundary conditions
        A[0, 0] = 1
        A[-1, -1] = 1

        B[0, 0] = 0
        B[-1, -1] = 0

        # vertical velocities = eigenvectors
        ev, w = scipy.linalg.eig(A, B)
        ev = np.real(ev)
        w = np.real(w)

        # horizontal velocities = central/forward/backward difference
        u = np.zeros(w.shape)
        u[1:-1, :] = (w[2:, :] - w[:-2, :]) / (z[2:] - z[:-2])[:, np.newaxis]
        u[0, :] = (w[1, :] - w[0, :]) / (z[1] - z[0])
        u[-1, :] = (w[-1, :] - w[-2, :]) / (z[-1] - z[-2])                        
        u = -u    # consistent with kundu
    
    elif method == 'kundu':
        N2l = (N2[:-2] +  N2[1:-1]) / 2
        N2u = (N2[1:-1] + N2[2:]) / 2

        # fill in matrices
        A[i, i+1] = 1 / (dzd * dzu * N2u)
        A[i, i] = - (1/dzd) * (1 / (dzl * N2l) + 1 / (dzu * N2u))
        A[i, i-1] = 1 / (dzd * dzl * N2l)

        B = np.diag(-np.ones(len(z)))

        # boundary conditions
        A[0, 1] = 1 / (z[1] - z[0])
        A[0, 0] = -1 / (z[1] - z[0])
        A[-1, -1] = 1 / (z[-1] - z[-2]) + N2[-1] / (4 * g)
        A[-1, -2] = -1 / (z[-1] - z[-2]) + N2[-2] / (4 * g)

        B[0, 0] = 0
        B[-1, -1] = 0

        # horizontal velocity = eigenvectors
        ev, u = scipy.linalg.eig(A, B)
        ev = np.real(ev)
        u = np.real(u)

        # vertical velocities
        w = np.zeros(u.shape)
        dz = np.diff(z)
        w[1:, :] = np.cumsum(0.5 * (u[1:, :] + u[:-1, :]) * dz[:, np.newaxis], axis=0)
        w = -w    # consistent with klinck

    # normalize
    u_norm = np.sqrt(np.trapezoid(u**2, x=z, axis=0))
    u = u / u_norm
    w_norm = np.sqrt(np.trapezoid(N2[:, np.newaxis] * (w**2), x=z, axis=0))
    w = w / w_norm

    # select positive eigenvalues
    mask = ev > 1e-10
    ev = ev[mask]
    u = u[:, mask]
    w = w[:, mask]

    # sort by eigenvalue
    i_sort = np.argsort(ev)
    ev = ev[i_sort]
    u = u[:, i_sort]
    w = w[:, i_sort]

    # invert depth
    u = u[::-1,:]
    w = w[::-1,:]

    if method == 'klinck':
        ev = ev[: nmodes]
        u = u[:, : nmodes]
        w = w[:, : nmodes]
    elif method == 'kundu':
        ev = ev[1: nmodes+1]
        u = u[:, 1: nmodes+1]
        w = w[:, 1: nmodes+1]

    # store as xarray DataArrays
    ev = xr.DataArray(ev, dims='mode', coords={'mode': np.arange(1, nmodes+1)}, name='eigenvalue')
    u = xr.DataArray(u[1:, :], dims=('depth', 'mode'), coords={'depth': depth, 'mode': np.arange(1, nmodes+1)}, name='velocity')
    w = xr.DataArray(w[1:, :], dims=('depth', 'mode'), coords={'depth': depth, 'mode': np.arange(1, nmodes+1)}, name='velocity')

    return ev, u, w


# ---------- Internal Seiche ---------- #

def modified_merian_equation(rho1, rho2, h1, h2, L):
    """
    Calculate internal seiche period for two-layer system.
    """
    raise NotImplementedError


def v1h1_mode(h1, h2, rho1, rho2, L):
    """
    Calculate V1H1 internal wave frequency/period.

    Parameters
    ----------
    h1 : float
        Surface layer depth [m].
    h2 : float
        Internal layer depth [m].
    rho1 : float
        Surface layer density [kg/m^3].
    rho2 : float
        Internal layer density [kg/m^3].
    L : float
        Characteristic length [m].

    Returns
    -------
    f : float
        V1H1 frequency [Hz].
    T : float
        V1H1 period [s].
    """
    g = 9.81

    # reduced gravity
    g_prime = g * (rho2 - rho1) / rho2

    # phase speed
    c = np.sqrt(g_prime * (h1 * h2) / (h1 + h2))

    # V1H1 period, frequency
    T = 2 * L / c
    f = 1/T

    return f, T