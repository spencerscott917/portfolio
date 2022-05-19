import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
import time
from multiprocessing import Pool
from model import *


base_dir ='/home/spencerscott/classes/cosmo/final_proj/data/IN_PAPER/'
folder_names = ['GGL003400-094434_1',  'GGL020928-100653_1',  'GGL100049+044553_1',  'GGL110920-000655_1',  'GGL130200-065613_1',  'GGL214705-012125_1',  'GGL231916-223942_1',
'GGL003402-094532_1',  'GGL030700''-071242_1',  'GGL103257+120139_1',  'GGL120439+014609_1',  'GGL134940+110621_1',  'GGL221230-241804_1',  'GGL235410+002339_1',
'GGL003402-094532_2',  'GGL100035+032254_1',  'GGL103439+110321_1',  'GGL120439+014609_2',  'GGL140313+140843_1',  'GGL223150+002627_1',  'GGL235410+002339_2'] 



# multiprocessing params
ncpus = 12 # number of processes to spawn

# mcmc params
ndim = 8 # number of free parameters
nwalkers = 300 
nsteps = 10000 

# free parameters in SAM for testing
xi0_guess = np.mean(xi)
eta0_guess = np.mean(eta)
theta_guess = 0
i_guess = 0
gamma_guess = 0
vmax_guess = np.median(np.abs(im[~np.isnan(im)]-np.mean(im[~np.isnan(im)])))
v0_guess =np.mean(im[~np.isnan(im)])
rt_guess= float(dat[0].header['R_T'])


# skeleton MCMC code
def log_likelihood(params, xi, eta, im, yerr):
    xi0, eta0, theta, i, gamma, vmax, v0, rt = params
    model = lens_model(xi, eta, xi0, eta0, theta, i, gamma, vmax, v0, rt)
    model = model[~np.isnan(im.T[::-1].flatten())]
    yerr = yerr.T[::-1][~np.isnan(im.T[::-1])].flatten()
    im = im.T[::-1][~np.isnan(im.T[::-1])].flatten()
    sigma2 = yerr ** 2
    # drop division by 0
    arg = (im - model) ** 2 / sigma2
    arg = arg[np.isfinite(arg)]
    return -0.5 * np.sum(arg)


def log_prior(params):
    xi0, eta0, theta, i, gamma, vmax, v0, rt = params
    # constraining parameters to solutions that make physical sense with
    # flat priors
    if ((0 < xi0) and
       (xi0 < im.shape[0]) and 
       (0 < eta0) and 
       (eta0 < im.shape[1]) and 
       (0 <= theta) and 
       (theta <= 6.28) and 
       (0 <= i) and 
       (i <= 6.28) and 
    #    (-0.03 <= gamma) and 
    #    (gamma <= 0.03) and
       (0 <= vmax) and 
       (vmax <= 5e3) and
       (0 <= v0) and 
       (v0 <= 1e5) and
       (0 <= rt) and 
       (rt <= 78)):
        return 0.0
    return -np.inf


def log_probability(params, xi, eta, im, yerr):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, xi, eta, im, yerr)


def mcmc(obj_name):
    filename = f"chains/chains_{obj_name}.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    p0 = [xi0_guess, eta0_guess, theta_guess, i_guess, gamma_guess, vmax_guess, v0_guess, rt_guess]

    # initialize walkers at different enough locations in parameter space
    pos = [
        p0 + np.array([5, 10, np.pi/4, np.pi/4, 0.001, 100, 1000, 5]) * np.random.randn(ndim) for i in range(nwalkers)
    ]
    with Pool(ncpus) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, backend=backend, args=(xi, eta, im, unc))
        start = time.time()
        sampler.run_mcmc(pos, nsteps)
        end = time.time()
    print(end - start)

    # # plotting
    # samples = sampler.chain[:, :, :].reshape((-1, ndim))
    # for xi0, eta0, theta, i, gamma, vmax, v0, rt in samples[np.random.randint(len(samples), size=5)]:
    #     smf = calc_sam(sn_reheat, sn_eject, kinetic_coupling, alpha, f_ICL)
    #     plt.plot(10**obs_smf[:,0], smf, alpha=0.3, label='SAM Samples')
    # plt.errorbar(10**obs_smf[:,0], 10**obs_smf[:,1], yerr=errors, label='Observed SMF')
    # plt.ylabel('SMF')
    # plt.xlabel('Halo Mass')
    # plt.legend()
    # plt.loglog()
    # plt.savefig('random_models.pdf')
    return


if __name__ == '__main__':
    mcmc(obj_id)
    # # commented out test code 
    # smf = calc_sam(sn_reheat_fraction, sn_eject_fraction, kinetic_coupling_to_cold_gas,
    #                alpha, f_ICL)
    # plt.plot(10**obs_smf[:,0], 10**smf)
    # plt.errorbar(10**obs_smf[:,0], 10**obs_smf[:,1], yerr=errors)
    # plt.loglog()
    # plt.show()