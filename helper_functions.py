import pandas as pd
import numpy as np

def display_summary(OIII_ratios, SII_ratios, temp, den):
    '''
    Function to display the summary of the results
    
    Parameters:
    
    OIII_ratios: list of float
        List of OIII 4363/5007+ ratios
    SII_ratios: list of float
        List of SII 6716/6731 ratios
    temp: list of float
        List of temperatures in K
    den: list of float
        List of electron densities in cm-3

    Returns:
        None
    '''
    for i in range(len(OIII_ratios)):
        print(f"OIII 4363/5007+ = {OIII_ratios[i]:.3f}, SII 6716/6731 = {SII_ratios[i]:.3f}, T = {temp[i]:.0f} K, n_e = {den[i]:.0f} cm-3")


def run_GetCritDensity(atom, K):
    
    return atom.getCritDensity(K)


def dist_getcritdensity(atom, K, return_type = 'array'):

    #making a two 2 array to hold the results of the critical densities
    #len(K) is how many temperatures we have
    #5 is the number of critical densities returned to use by the function
    
    store_crit_density = np.ones((len(K), 5))

    for idx, temp in enumerate(K):
        if np.isfinite(temp):
            crit_density = run_GetCritDensity(atom, temp)
            store_crit_density[idx] = crit_density
        else:
            store_crit_density[idx] = np.nan

    if return_type == 'df':
        store_crit_density = pd.DataFrame(store_crit_density, columns = ['N_crit_1', 'N_crit_2', 'N_crit_3', 'N_crit_4', 'N_crit_5'])
        store_crit_density['Temperature'] = K

    return store_crit_density


def convert_int_relative_to_hbeta(int_ion, int_hbeta):

    return (int_ion/int_hbeta) * 100

def ionic_abundance(ion, int, tem, den, wave):
    
    ion_abu = ion.getIonAbundance(int_ratio = int, tem = tem, den = den, wave = wave)
    
    return ion_abu
