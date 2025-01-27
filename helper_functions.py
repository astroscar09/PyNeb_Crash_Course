import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pyneb as pn

DataFileDict = {'H1': {'rec': 'h_i_rec_SH95.fits'},
                'O2': {'atom': 'o_ii_atom_FFT04.dat', 'coll': 'o_ii_coll_Kal09.dat'},
                'S2': {'coll': 's_ii_coll_TZ10.dat'}}

pn.atomicData.setDataFileDict(DataFileDict)

diag = pn.Diagnostics()
diag.addDiag('[SII] 6716/6731', ('S2', 'L(6716)/L(6731)', 'RMS([E(6716), E(6731)])'))

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


def compute_cross_temden(inputs):
    OIII_ratio, SII_ratio = inputs
    return diag.getCrossTemDen('[OIII] 4363/5007+', '[SII] 6716/6731', OIII_ratio, SII_ratio)


def cross_temden(OIII_ratios, SII_ratios, multiprocess = False):

    '''
    This function uses the PyNeb getCrossTemDen function to compute the temperature and density
    This function is to be used when the OIII and SII ratios are large arrays

    Example run time with 10,000 random OIII and SII ratios: 
    with multiprocess = True: 5 minutes, >16 minutes without multiprocess

    Parameters:
    OIII_ratios: list of float
        List of OIII 4363/5007+ ratios
    
    SII_ratios: list of float
        List of SII 6716/6731 ratios
    
    multiprocess: bool, optional
        Whether to use multiprocessing
    
    Returns:
    
    temp: list of float
        List of temperatures in K
    
    den: list of float
        List of electron densities in cm^-3
    '''

    if multiprocess:
        
        temden_inputs = [(OIII, SII) for OIII, SII in zip(OIII_ratios, SII_ratios)]

        # Parallel computation
        with Pool(4) as p:  # Adjust the number of processes as needed
            results = p.map(compute_cross_temden, temden_inputs)
        
        temps, dens = zip(*results)
        
        temp, den = list(temps), list(dens)
    
    else:
        temp, den = diag.getCrossTemDen('[OIII] 4363/5007+', '[SII] 6716/6731', OIII_ratios, SII_ratios)

    return temp, den


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


def get_ion_density_limits(ion, species, temp = 1e4):

    '''
    Function to get the lower and upper density limits for a given ion

    Parameters:
    
    ion: Pyneb Ion object
        Ion object from Pyneb
    
    species: str
        Species of the ion [OIII, SII supported atm]
    
    temp: float, optional
        Temperature in Kelvin
    
    Returns:
    
    den_limits: array of float
        Array of lower and upper density limits
    '''
    
    map_species = {'OIII': 'L(4363)/(L(5007)+L(4959))',
                   'SII': 'L(6716)/L(6731)',            #we need to add this one in through addDiag I think
                    }
    
    pyneb_input = map_species[species]

    low_den_lim = ion.getLowDensRatio(to_eval = pyneb_input, temp = temp)
    high_den_limit = ion.getLowHighRatio(to_eval = pyneb_input, temp = temp)

    den_limits = np.array([low_den_lim, high_den_limit])

    if not np.isfinite([den_limits]).all():
        print('WARNING')
        print('-------------------------------------------')
        print('At least one of the density limits is a NaN')
        print(den_limits)
        print('-------------------------------------------')
        

    return den_limits

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

    mask = np.isnan(temp_arr)

    temp_arr[mask] = value
    
    return temp_arr

def plot_distribution(data, ax, color):

    '''
    Function to plot a distribution of data

    Parameters:
    data: array of float
        Data to plot
    color: str
        Color of the plot
    label: str
        Label of the plot
    ax: matplotlib axis object, optional
        Axis to plot on

    Returns:
        None
    '''

    
    ax.hist(data, bins = 35, color = color, alpha = 0.5)
    
def compute_OIII_metallicity(O3, OIII_4363, OIII_4969, OIII_5007, Hbeta, 
                             temp, den, ICF = 1, plot = None):
    
    '''
    Function to compute the metallicity using the OIII 4363, 4969, 5007 lines

    Parameters:
    
    O3: Pyneb Ion object
        Ion object from Pyneb
    
    OIII_4363: float or array of floats
        Intensity of the OIII 4363 line
    
    OIII_4969: float or array of floats
        Intensity of the OIII 4969 line
    
    OIII_5007: float or array of floats
        Intensity of the OIII 5007 line
    
    Hbeta: float or array of floats
        Intensity of the Hbeta line
    
    temp: float or array of floats
        Temperature in Kelvin
    
    den: float or array of floats
        Electron density in cm^-3
    
    plot: bool, optional
        Whether to plot the distribution of the metallicities
    
    Returns:
    l16: float
        16th percentile of the distribution
    
    med: float
        Median of the distribution
    
    u84: float
        84th percentile of the distribution
    
    total_OIII_abundance: array of floats
        Distribution of metallicities
    '''

    #converting the OIII lines to intensities relative to Hbeta * 100 for PyNeb
    OIII5007_int = convert_int_relative_to_hbeta(OIII_5007, Hbeta)
    OIII4969_int = convert_int_relative_to_hbeta(OIII_4969, Hbeta)
    OIII4363_int = convert_int_relative_to_hbeta(OIII_4363, Hbeta)

    #getting the ionic abundances for the OIII lines
    OIII_5007_ion_abundance = ionic_abundance(O3, OIII5007_int, temp, den, 5007)
    OIII_4969_ion_abundance = ionic_abundance(O3, OIII4969_int, temp, den, 4969)
    OIII_4363_ion_abundance = ionic_abundance(O3, OIII4363_int, temp, den, 4363)

    # we may need to weight the sum by some factor so that dominant terms are not overrepresented
    
    #w_OIII5007 = OIII5007_int * OIII_5007_ion_abundance
    #w_OIII4969 = OIII4969_int * OIII_4969_ion_abundance
    #w_OIII4363 = OIII4363_int * OIII_4363_ion_abundance

    #weight_sum = np.sum([w_OIII5007, w_OIII4969, w_OIII4363], axis = 0)
    #sum_int = np.sum([OIII5007_int, OIII4969_int, OIII4363_int], axis = 0)

    #computing the total OIII abundance, summing across the columns resulting in a distribution of
    #Metallicites for each of the temperatures and densities provided
    OIII_abu = np.average([OIII_5007_ion_abundance, 
                           OIII_4969_ion_abundance, 
                           OIII_4363_ion_abundance], 
                           weights = [OIII5007_int, 
                                      OIII4969_int, 
                                      OIII4363_int], axis = 0)
    
    #np.sum([OIII_5007_ion_abundance, 
    #        OIII_4969_ion_abundance, 
    #        OIII_4363_ion_abundance], 
    #        axis = 0)

    #after this we may need to apply an ICF for OIII to get the total O abundance
    #Assume ICF(O) = 1

    O_abund = OIII_abu * ICF

    
    #converting this to 12+log(O/H) using the relation from the PyNeb documentation
    metallicity = 12 + np.log10(O_abund)

    l16, med, u84 = np.percentile(metallicity, [16, 50, 84])

    if plot:
        
        fig, ax = plt.subplots()
        
        plot_distribution(metallicity, ax, 'purple')
        
        ax.set_xlabel('12 + log(O/H)', fontsize = 15)
        ax.set_ylabel('Counts', fontsize = 15)
        
        ax.axvline(l16, color = 'red', linestyle = '-', label = f'16th Percentile')
        ax.axvline(med, color = 'black', linestyle = '--', label = f'Median')
        ax.axvline(u84, color = 'red', linestyle = '-', label = f'84th Percentile')
        
        ax.legend()
        plt.show()


    return l16, med, u84, metallicity

