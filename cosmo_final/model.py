import numpy as np
from astropy.io import fits

fname='/home/spencerscott/classes/cosmo/final_proj/data/IN_PAPER/GGL223150+002627_1/GGL223151+0026.fits'
obj_id = 'id13'
dat = fits.open(fname)
im = dat[0].data
delta = float(dat[0].header['DETLA'])
unc = dat[1].data
im[unc>50] = np.nan
# unc = unc.flatten()[~np.isnan(im)]

# defining detector coordinates based on image shape to use for transforms in modeling
xi, eta = np.meshgrid(list(range(im.shape[0])),list(range(im.shape[1])), indexing='xy')
xi = xi.flatten()
eta = eta.flatten()


def R_inv(theta):
    """The inverse of R(theta) from Equation 8 in Gurri+2020"""
    costh = np.cos(theta)
    sinth = np.sin(theta)
    return np.array([[costh, sinth],[-sinth, costh]])


def I_inv(i):
    """The inverse of I(i) from Equation 8 in Gurri+2020"""
    return np.array([[1,0],[0, 1/np.sin(i)]])


def Aprime(gamma, delta):
    """A' in equation 9, with kappa set to 0 as per the last paragraph on the page"""
    twodelt = 2 * delta
    gamcos2delt = gamma * np.cos(twodelt) 
    gamsin2delt = gamma * np.sin(twodelt)
    return np.array([[1 - gamcos2delt, -gamsin2delt],[-gamsin2delt, 1 + gamcos2delt]])


def gal_V(x, y, vmax, v0, rt):
    """V(x,y) in galactocentric coordinates in eqn 6"""
    R = np.sqrt((x)**2 + (y)**2)
    omega = np.arctan2(y,x)
    return 2 * vmax / np.pi * np.arctan2(np.abs(R),rt) * np.sin(omega) + v0


def det_to_gal_coords(dat, theta, i, gamma, delta):
    return np.dot(I_inv(i),np.dot(R_inv(theta),np.dot(Aprime(gamma, delta), dat)))


def lens_model(xi, eta, xi0, eta0, theta, i, gamma, vmax, v0, rt):
    """Combining above transforms to give full model that runs through MCMC"""
    mat = np.array([xi-xi0,eta-eta0])

    x,y = det_to_gal_coords(mat, theta, i, gamma, delta)
    fit = gal_V(x, y, vmax, v0, rt)
    z = fit.flatten()
    z[np.isnan(im.T[::-1].flatten())] = np.nan
    return z


# z = z.reshape(im.T[::-1].shape[0],im.T[::-1].shape[1])