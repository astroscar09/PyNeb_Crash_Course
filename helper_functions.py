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

    '''
    Runs the Pyneb GetCritDensity function for a given atom and temperature
    (Assumes a single temperature is passed)

    Parameters:
    atom: Pyneb Atom object
        Atom object from Pyneb
    K: float The temperature in Kelvin

    Returns:
    crit_density: list of float
        List of critical densities for the given temperature

    '''
    
    return atom.getCritDensity(K)


def dist_getcritdensity(atom, K, return_type = 'array'):

    '''
    Function to get the critical densities for a range of temperatures 
    this function can handle NaN values in the temperature array
    it does so by appending the critical densities as NaN values

    Parameters:
    atom: Pyneb Atom object
        Atom object from Pyneb
    K: list of floats, temperatures in Kelvin
    return_type: str, optional
        Type of the return, 'array' or 'df'

    Returns:
    store_crit_density: array or DataFrame
        Array or DataFrame of critical densities for the given temperatures
    '''

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

    '''
    Function to convert the intensity of an ion relative to Hbeta *100 as per PyNeb requirements

    Parameters:
    int_ion: float
        Intensity of the ion
    int_hbeta: float
        Intesnity of Hbeta

    Returns:
    int_ratio: float
        Intensity of the ion relative to Hbeta * 100
    '''

    return (int_ion/int_hbeta) * 100

def ionic_abundance(ion, int, tem, den, wave):

    '''
    Function that gets the ionic abundance for a given ion

    Parameters:
    ion: Pyneb Ion object
        Ion object from Pyneb
    int: float Intesnity of the ion relative to Hbeta * 100
    tem: float Temperature in Kelvin
    den: float Electron density in cm-3
    wave: float Wavelength of the ion

    Returns:
    ion_abu: float
        Ionic abundance of the ion

    '''
    
    ion_abu = ion.getIonAbundance(int_ratio = int, tem = tem, den = den, wave = wave)
    
    return ion_abu

def replace_NaNs(arr, value):

    '''
    Function to assign a value to NaN values in an array

    Parameters:
    temp: list of float
        array of values
    value: float
        Value to replace NaN values with

    Returns:
    temp_arr: list of float
        array with NaN values replaced
    '''
    
    temp_arr = arr.copy() #added this copy to not overwrite the original input array

    for idx, temp in enumerate(temp_arr):
        if np.isnan(temp):
            temp_arr[idx] = value
    
    return temp_arr