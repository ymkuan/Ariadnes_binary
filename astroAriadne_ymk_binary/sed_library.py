"""sed_library contain the model, prior and likelihood to be used."""

import numba as nb
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from extinction import apply

from .utils import get_noise_name
from .config import gridsdir


def build_params(theta, flux, flux_e, filts, coordinator, fixed, use_norm_1, use_norm_2):
    """Build the parameter vector that goes into the model."""
    params = np.zeros(len(coordinator))
    if use_norm_1 and use_norm_2:
        order = np.array(['teff1', 'logg1', 'z1', 'teff2', 'logg2', 'z2', 'norm1', 'norm2', 'Av'])
    elif use_norm_1 and not use_norm_2:
        order = np.array(['teff1', 'logg1', 'z1', 'teff2', 'logg2', 'z2', 'dist', 'norm1', 'rad2', 'Av'])
    elif not use_norm_1 and use_norm_2:
        order = np.array(['teff1', 'logg1', 'z1', 'teff2', 'logg2', 'z2', 'dist', 'rad1', 'norm2', 'Av'])
    elif not use_norm_1 and not use_norm_2:
        order = np.array(['teff1', 'logg1', 'z1', 'teff2', 'logg2', 'z2', 'dist', 'rad1', 'rad2', 'Av'])

    for filt, flx, flx_e in zip(filts, flux, flux_e):
        p_ = get_noise_name(filt) + '_noise'
        order = np.append(order, p_)
    i = 0
    for j, k in enumerate(order):
        params[j] = theta[i] if not coordinator[j] else fixed[j]
        if not coordinator[j]:
            i += 1
    return params


def get_interpolated_flux(temp, logg, z, filts, interpolator):
    """Interpolate the grid of fluxes in a given teff, logg and z.

    Parameters
    ----------
    temp: float
        The effective temperature.
    logg: float
        The superficial gravity.
    z: float
        The metallicity.
    filts: str
        The desired filter.

    Returns
    -------
    flux : float
        The interpolated flux at temp, logg, z for filter filt.

    """
    values = (logg, temp, z)
    flux = interpolator(values, filts)
    return flux


def model_grid(theta, filts, wave, interpolator_1, interpolator_2, use_norm_1, 
               use_norm_2, av_law, model_1, model_2):
    """Return the model grid in the selected filters.

    Parameters:
    -----------
    theta : array_like
        The parameters of the fit: teff, logg, z, radius, distance
    star : Star
        The Star object containing all relevant information regarding the star.
    interpolators : dict
        A dictionary with the interpolated grid.
    use_norm : bool
        False for a full fit  (including radius and distance). True to fit
        for a normalization constant instead.

    Returns
    -------
    model : dict
        A dictionary whose keys are the filters and the values are the
        interpolated fluxes

    """
    Rv = 3.1  # For extinction.

    if use_norm_1 and use_norm_2:
        teff1, logg1, z1, teff2, logg2, z2, norm1, norm2, Av = theta[:9]
    elif use_norm_1 and not use_norm_2:
        teff1, logg1, z1, teff2, logg2, z2, dist, norm1, rad2, Av = theta[:10]
    elif not use_norm_1 and use_norm_2:
        teff1, logg1, z1, teff2, logg2, z2, dist, rad1, norm2, Av = theta[:10]
    elif not use_norm_1 and not use_norm_2:
        teff1, logg1, z1, teff2, logg2, z2, dist, rad1, rad2, Av = theta[:10]
    
    if not use_norm_1 or not use_norm_2:
        dist *= 4.435e+7  # Transform from pc to solRad

    flux_1 = get_interpolated_flux(teff1, logg1, z1, filts, interpolator_1)
    flux_2 = get_interpolated_flux(teff2, logg2, z2, filts, interpolator_2)

    if np.any(np.isnan(flux_1)) and (model_1.lower() == 'koesterbb' or model_1.lower() == 'koester' or model_1.lower() == 'tmap'):
        flux_1 = check_flux_nan(flux_1, teff1, logg1, filts, model_1)
    if np.any(np.isnan(flux_2)) and (model_2.lower() == 'koesterbb' or model_2.lower() == 'koester'or model_2.lower() == 'tmap'):
        flux_2 = check_flux_nan(flux_2, teff2, logg2, filts, model_2)


    wav = wave * 1e4
    ext = av_law(wav, Av, Rv)

    if use_norm_1:
        model_1 = apply(ext, flux_1) * norm1
    if not use_norm_1:
        model_1 = apply(ext, flux_1) * (rad1 / dist) ** 2

    if use_norm_2:
        model_2 = apply(ext, flux_2) * norm2
    if not use_norm_2:
        model_2 = apply(ext, flux_2) * (rad2 / dist) ** 2
    model = model_1 + model_2

    return model


def get_residuals(theta, flux, flux_er, wave, filts, interpolator_1, interpolator_2, 
                  use_norm_1, use_norm_2, av_law, model_1, model_2):
    """Calculate residuals of the model."""
    model = model_grid(theta, filts, wave, interpolator_1, interpolator_2, 
                       use_norm_1, use_norm_2, av_law, model_1, model_2)
    
    if use_norm_1 and use_norm_2:
        start = 9
    else:
        start = 10
    inflation = theta[start:]
    residuals = flux - model


    errs = np.sqrt(flux_er ** 2 + inflation ** 2)
    return residuals, errs


def log_likelihood(theta, flux, flux_er, wave, filts, interpolator_1, 
                   interpolator_2, use_norm_1, use_norm_2, av_law, model_1, model_2):
    """Calculate log likelihood of the model."""
    res, ers = get_residuals(theta, flux, flux_er, wave, filts, interpolator_1, 
                             interpolator_2, use_norm_1, use_norm_2, av_law, model_1, model_2)

    lnl = fast_loglik(res, ers)

    if not np.isfinite(lnl):
        return -1e300

    return -.5 * lnl


@nb.njit
def fast_loglik(res, ers):
    ers2 = ers ** 2
    c = np.log(2 * np.pi * ers2)
    lnl = (c + (res ** 2 / ers2)).sum()
    return lnl


def prior_transform_dynesty(u, flux, flux_er, filts, prior_dict, coordinator,
                            use_norm_1, use_norm_2):
    """Transform the prior from the unit cube to the parameter space."""
    u2 = np.array(u)
    # Declare order of parameters.
    if not use_norm_1 and not use_norm_2:
        order = np.array([
            'teff1', 'logg1', 'z1', 'teff2', 'logg2', 'z2',
            'dist', 'rad1', 'rad2', 'Av'])
        
    elif not use_norm_1 and use_norm_2:
        order = np.array([
            'teff1', 'logg1', 'z1', 'teff2', 'logg2', 'z2',
            'dist', 'rad1', 'norm2', 'Av'])
            
    elif use_norm_1 and not use_norm_2:
        order = np.array([
            'teff1', 'logg1', 'z1', 'teff2', 'logg2', 'z2',
            'dist', 'norm1', 'rad2', 'Av'])
        
    elif use_norm_1 and use_norm_2:
        order = np.array(['teff1', 'logg1', 'z1', 'teff2', 'logg2', 'z2', 'norm1', 'norm2', 'Av'])
            

    for filt, flx, flx_e in zip(filts, flux, flux_er):
        p_ = get_noise_name(filt) + '_noise'
        order = np.append(order, p_)

    i = 0
    for fixed, par in zip(coordinator, order):
        if fixed:
            continue
        if par == 'logg1':
            try:
                u2[i] = prior_dict['logg1'](u2[i])
            except TypeError:
                u2[i] = prior_dict['logg1'].ppf(u2[i])
            i += 1
            continue
        if par == 'logg2':
            try:
                u2[i] = prior_dict['logg2'](u2[i])
            except TypeError:
                u2[i] = prior_dict['logg2'].ppf(u2[i])
            i += 1
            continue

        if par == 'teff1':
            try:
                u2[i] = prior_dict['teff1'](u2[i])
            except TypeError:
                u2[i] = prior_dict['teff1'].ppf(u2[i])
            i += 1
            continue
        if par == 'teff2':
            try:
                u2[i] = prior_dict['teff2'](u2[i])
            except TypeError:
                u2[i] = prior_dict['teff2'].ppf(u2[i])
            i += 1
            continue
        u2[i] = prior_dict[par].ppf(u2[i])
        i += 1
    return u2


def prior_transform_multinest(u, flux, flux_er, filts, prior_dict, coordinator,
                              use_norm):
    """Transform the prior from the unit cube to the parameter space."""
    if use_norm:
        order = np.array(['teff', 'logg', 'z', 'norm', 'Av'])
    else:
        order = np.array(
            ['teff', 'logg', 'z', 'dist', 'rad', 'Av']
        )

    for filt, flx, flx_e in zip(filts, flux, flux_er):
        p_ = get_noise_name(filt) + '_noise'
        order = np.append(order, p_)

    i = 0
    for fixed, par in zip(coordinator, order):
        if fixed:
            continue
        if par == 'logg':
            try:
                u[i] = prior_dict['logg'](u[i])
            except TypeError:
                u[i] = prior_dict['logg'].ppf(u[i])
            i += 1
            continue
        if par == 'teff':
            try:
                u[i] = prior_dict['teff'](u[i])
            except TypeError:
                u[i] = prior_dict['teff'].ppf(u[i])
            i += 1
            continue
        u[i] = prior_dict[par].ppf(u[i])
        i += 1
    pass
def check_flux_nan(flux, teff, logg, filts, model):
    '''
    发现koesterWD的模型，在温度40000K时，插值出现nan的情况
    于是这个代码就是为了检验是否有nan值有的话，替换掉
    '''
    nan_indices = np.where(np.isnan(flux))[0]
    
    for dex in nan_indices:
        filt = filts[dex]
        if model.lower() == 'koesterbb':
            with open(gridsdir + '/KoesterBB_DF.pkl', 'rb') as wd:
                wd_pkl = pd.read_pickle(wd)
            points_2d = wd_pkl.index.droplevel('[Fe/H]').to_frame(index=False).values  # 只取 logg 和 Teff
            values = wd_pkl[filt].values  # 插值目标值
            new_points_2d = [logg, teff]
            
            # 进行二维插值
            flux_interp = griddata(points_2d, values, new_points_2d, method='linear')
            flux[dex] = flux_interp

        if model.lower() == 'koester':
            with open(gridsdir + '/Koester_DF.pkl', 'rb') as wd:
                wd_pkl = pd.read_pickle(wd)
            points_2d = wd_pkl.index.droplevel('[Fe/H]').to_frame(index=False).values  # 只取 logg 和 Teff
            values = wd_pkl[filt].values  # 插值目标值
            new_points_2d = [logg, teff]
            
            # 进行二维插值
            flux_interp = griddata(points_2d, values, new_points_2d, method='linear')
            flux[dex] = flux_interp

        if model.lower() == 'tmap':
            with open(gridsdir + '/Tmap_DF.pkl', 'rb') as sdb:
                sdb_pkl = pd.read_pickle(sdb)
            points_2d = sdb_pkl.index.droplevel('[Fe/H]').to_frame(index=False).values  # 只取 logg 和 Teff
            values = sdb_pkl[filt].values  # 插值目标值
            new_points_2d = [logg, teff]
            
            # 进行二维插值
            flux_interp = griddata(points_2d, values, new_points_2d, method='linear')
            flux[dex] = flux_interp
    return flux





