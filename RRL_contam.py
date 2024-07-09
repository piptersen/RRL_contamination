import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as k
from scipy.integrate import quad



def rest_freq(n):
    '''
    Calculates H transition frequency in MHz
    ------------------
    n            (int)   : Transition level
    ------------------
    return [MHz] (float) : Frequency of n transition
    '''
    return(((1 / (n)**2) - (1 / (n+1)**2)) * 1.1e5 * 3e10 * 1e-6)



def freq_z(nu_emit, z):
    '''
    Calculates cosmological redshift of ground state H frequency
    ------------------
    nu_emit (float) : Ground state frequency of transition
    z       (array) : Redshift space of 21cm emission
    ------------------
    return  (array) : Redshifted frequency of H transition
    '''
    return(nu_emit/(1+z))

def contam21cm_z(nu_obs):
    return((1420 / nu_obs) - 1)


def beta_n(nu, b, n, T):
    '''
    Calculates departure coefficient from LTE, beta_n, of HII region
    ------------------
    nu     (array) : Frequency of n transitions
    b      (array) : Departure coefficient b, from Table
    n      (int)   : Transition levels of departure coefficients, b
    T      (float) : Temperature associated with departure coefficient simulation, b
    ------------------
    return (array) : Departure coefficient, beta_n for each transition n
    '''
    h = 6.6261e-27
    k = 1.3807e-16
    bn = b[n]
    bn_p = b[n+1]
    numer = (1 - ((bn_p / bn)*np.exp(-((h*nu) / (k*T)))))
    den = (1 - (np.exp(-((h*nu) / (k*T)))))
    return (numer / den)



def GPP_fit(b, T, plot_GPFit=False):
    '''
    Performs Gaussian Process fit to combined departure coefficient, bn*beta_n
    ------------------
    b          (array)   : Departure coefficient b, from Table
    T          (float)   : Temperature associated with departure coefficient simulation, b
    plot_GPFit (boolean) : If interested in plotting the gaussian process fit w/ the raw data from Table
    ------------------
    return     (array)   : Interpolated function of bn*beta_n for each n value 
    '''
    n_full = np.arange(2, len(b)-1, 1)
    nu_range = rest_freq(n_full)
    beta_range = beta_n(nu_range, b, n_full, 1e4)
    bnbetan = b[n_full] * beta_range

    bnbetan[180:]=1
    bbeta_shaped = bnbetan.reshape(-1,1)
    n_shaped = n_full.reshape(-1,1)

    kernel = np.var(bbeta_shaped) * k.RBF(length_scale=10.0)
    
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=90, alpha=1e5)
    gp.fit(n_shaped, bbeta_shaped)
    optimizedParams = gp.kernel_

    bbeta_model, fit_unc = gp.predict(n_shaped, return_std=True)
    if plot_GPFit == True:
        plt.plot(n_full, bbeta_model, color = 'red')
        plt.xlabel('n Transition')
        plt.ylabel(r'$b_n \beta_n$')
        plt.title('GMM Departure Coefficient Fit')
        plt.scatter(n_full, bnbetan, s = 15)
    return(bbeta_model)



def tau_test(T, EM, n, bbeta):
    '''
    Calculates optical depth of each transition, n
    ------------------
    T            (float) : Temperature associated with departure coefficient simulation, b
    EM           (float) : Emission Measure of HII region
    n            (array) : Transition levels of departure coefficients, b
    bbeta        (array) : Interpolated bn*beta_n values
    ------------------
    return [MHz] (array) : Optical depth of each transition, n
    '''
    chi = 1.58e5 / (T * ((n+2)**2))
    return(2.046e6 * np.exp(chi)* (T**(-5/2)) * EM * bbeta * 1e-6)



def H_z_calc(z):
    '''
    Function calculating the Hubble parameter w/ redshift dependence
    ---------------------------------------
    z                       (array) : redshift
    ---------------------------------------
    return [km / sec / Mpc] (array) : Hubble Constant at each redshift value
    '''
    return(70 * np.sqrt(0.73 + 0.27*(1+z)**3 + 8.24e-5*(1+z)**4))



def madauFit(z):
    '''
    Calculates the fit for SFR from redshift (Madau & Dickinson 2014)
    -----------------------------------
    z                                 (array) : Redshift 
    -----------------------------------
    returns [Solar Mass / yr / Mpc^3] (array) : Star formation rate Density
    '''
    return (0.015 * (1 + z)**2.7 / (1 + ((1 + z)/2.9)**5.6))


def delX_new(z):
    '''
    Integrand in delta_x calculation to calculate distance between 21cm emitter and RRL emitter
    ---------------------------------------
    z            (float) : Redshift where line is emitted
    ---------------------------------------
    return [Mpc] (float) : Distance between redshift z and observer
    '''
    c = 3e5
    return(c / H_z_calc(z))

def delX(nu_RRL, nu_21, z):
    '''
    Calculates the distance between 21cm emitter and RRL emitter
    ---------------------------------------
    nu_RRL       (float) : Rest frame frequency of RRL emission
    nu_21cm      (float) : Rest frame frequency of 21cm emission
    z            (array) : Redshift range where RRL would be observable
    ---------------------------------------
    return [Mpc] (array) : Distance between RRL emitter and 21cm emitter at each redshift z
    '''
    c = 3e5
    a = (1+z)**(-1)
    H = H_z_calc(z)
    return (((nu_RRL - nu_21) * c) / (a * H * nu_21))



def meanT_shifted(E_SFR, tau_L, N_HII, nu_emit, linewidth, bandwidth, z):
    '''
    Calculates the Line temperature of each RRL at the associated redshift. This function is only passed those RRLs (and associated redshifts) that are observable by 
    the 21cm observing experiment.
    ---------------------------------------
    E_SFR     [Solar Mass / yr / Mpc^3] (float) : Comoving Star Formation Rate
    tau_L                               (float) : Optical Depth of RRL in HII region
    N_HII                               (float) : Number of HII obscuring regions w/ optical depth, tau_L
    nu_emit   [MHz]                     (float) : Rest frame frequency of RRL
    linewidth [km / s]                  (float) : Linewidth of RRL, produced by HII region
    bandwidth [MHz]                     (float) : Bandwidth of 21cm observation
    z                                   (array) : Redshift where RRL is observable
    --------------------------------------
    return    [K]                       (array) : Line temperature of particular RRL
    '''
    E_SFR = madauFit(z)
    T_L = 3.9e-4 * (E_SFR / 0.1) * ((tau_L * N_HII) / 0.1) * (nu_emit / 150)**(-2.8) * (linewidth / 20) * (bandwidth / 1)**(-1) * ((1+z)/3)**(-0.5)
    return(T_L)



def T_RRL(T_n, deltaX, z):
    '''
    Calculates the distance between 21cm emitter and RRL emitter
    ---------------------------------------
    T_n    [K]   (float) : Line temperature of RRL
    deltaX [Mpc] (float) : Distance between RRL emitter and 21cm emitter
    ---------------------------------------
    return [K]   (array) : Contamination of 21cm emission
    '''
    b_RRL = 3 / (1+z)
    k = 2*np.pi / 150
    return(T_n * b_RRL * np.exp(np.imag(k*deltaX)))


def BB(nu, T):
    h = 6.6261e-27
    c = 3e5
    k = 1.3807e-16
    return(((2*h*nu**3) / c**2) * (1 / (np.exp((h*nu)/(k*T)) - 1)))


def frequency_contamination(freq_range, contam_ind, nu_stack):
    '''
    Optional function for breaking contamination by frequency bins and plotting
    ---------------------------------------
    freq_range [MHz] (array)                        : Frequency band of contamination interest
    contam_ind [K]   (list of lists)                : Full temperature contamination of 21cm emission (outer list is n transition, inner list is contamination by redshift)
    nu_stack   [MHz] (list of lists, as contam_ind) : Frequency associated with contam_ind
    ---------------------------------------
    return : None
    '''
    constack = list(itertools.chain.from_iterable(contam_ind))
    fstack = list(itertools.chain.from_iterable(nu_stack))
    hist, freq_range = np.histogram(fstack, bins=freq_range, weights=constack)
    fig, ax = plt.subplots()
    #ax.plot(freq_range[:-1], (BB(freq_range[:-1], 5e-9)), color = 'red')
    ax.plot(freq_range[:-1], -hist)
    #plt.axvline(x = 200.66, color = 'red')
    #plt.axvline(x = 204.33, color = 'red')
    #plt.axvline(x = 208.0, color = 'red')
    ax.set_xlabel(r'$\nu$')
    ax.set_ylabel(r'$T_{RRL}$')
    #ax.fill_between(fstack, min(-hist), max(-hist), alpha=0.2, color = 'orange')
    ax.set_title(f'21cm Contamination from Frequency Band {freq_range[0]}-{freq_range[-1]} MHz')
    print('Contamination of Frequency Band: ', np.absolute(np.sum(hist)), ' K')

def n_contamination(n_band, contam_ind, n_min):
    '''
    Optional function for breaking contamination by frequency bins and plotting
    ---------------------------------------
    n_band         (array)         : Transition band (n) of contamination interest
    contam_ind [K] (list of lists) : Full temperature contamination of 21cm emission (outer list is n transition, inner list is contamination by redshift)
    n_min          (int)           : Minimum n value of RRL contamination 
    ---------------------------------------
    return                         : None
    '''
    summed_list = np.absolute([sum(sublist) for sublist in contam_ind[n_band[0]-n_min:n_band[-1]-n_min]])
    fig, ax = plt.subplots()
    ax.scatter(n_band[:-1], summed_list)
    ax.set_title(f'21cm Contamination from Transition Band {n_band[0]} - {n_band[-1]}')
    ax.set_ylabel(r'$T_{RRL}$')
    ax.set_xlabel('n Transition')
    print('Contamination of n Band: ', np.absolute(np.sum(summed_list)), ' K')

def line_contamination(z_min, z_max, n_min, n_max, HI_freq_min, HI_freq_max, EM, T, E_SFR, N_HII, linewidth_unshifted, bandwidth, data, nu_band_min=400, nu_band_max=1420, n_band_min = 100, n_band_max = 450, plot_GPFit = False, plot_freqContam = False, plot_nContam = False):
    '''
    Wrapper function performing calculation of 21cm contamination by RRL
    ---------------------------------------
    z_min                                     (float)   : Minimum redshift of contamination interest
    z_max                                     (float)   : Maximum redshift of contamination interest
    n_min                                     (float)   : Minimum transition of RRL (1 < n_min < 450) (from bn values)
    n_max                                     (float)   : Maximum transition of RRL (n_min < n_max < 450) (from bn values)
    HI_freq_min     [MHz]                     (float)   : Minimum frequency of 21cm observing band 
    HI_freq_max     [MHz]                     (float)   : Maximum frequency of 21cm observing band 
    EM                                        (float)   : Emission Measure of HII region
    T               [K]                       (float)   : Temperature of HII region (defined by 'data')
    E_SFR           [Solar Mass / yr / Mpc^3] (float)   : Comoving Star Formation Rate
    N_HII                                     (float)   : Number of HII obscuring regions w/ optical depth, tau_L
    linewidth       [km / s]                  (float)   : Linewidth of RRL, produced by HII region
    bandwidth       [MHz]                     (float)   : Bandwidth of 21cm observation
    data                                      (string)  : Name of file containing table of bn values
    nu_band_min     [default = 400]           (float)   : (optional) Lower bound of frequency contamination band
    nu_band_max     [default = 1420]          (float)   : (optional) Upper bound of frequency contamination band
    n_band_min      [default = 100]           (int)     : (optional) Lower bound of transition contamination band
    n_band_max      [default = 450]           (int)     : (optional) Upper bound of transition contamination band
    plot_GPFit      [default = False]         (boolean) : (optional) If interested in plotting the gaussian process fit w/ the raw data from Table
    plot_freqContam [default = False]         (boolean) : (optional) Calculates and plots contamination by given frequency band
    plot_nContam    [default = False]         (boolean) : (optional) Calculates and plots contamination by given n transition band
    ---------------------------------------
    return          [K]                       (float)   : Total 21cm contamination by all Radio Recombination Lines
    '''
    z_space = np.linspace(z_min, z_max, 10_000)
    n_range = np.arange(n_min, n_max, 1)

    b = data[1::2]

    bbeta_new = GPP_fit(b, T, plot_GPFit)
    tau_L = tau_test(T, EM, n_range, bbeta_new[n_range])
    
    contam_ind = []
    nu_stack = []
    for i, n in enumerate(n_range):
        #For specific transition, the rest frame frequency
        nu_emitted = rest_freq(n)
        
        #For that transition, the frequency it is observable at at all redshifts post-reionization
        nu_shifted = freq_z(nu_emitted, z_space)
        
        #Mask that cuts out frequencies that will be redshifted out of observing window (won't contaminate 21cm)
        nu_obs_mask = (nu_shifted > HI_freq_min) & (nu_shifted < HI_freq_max)
        
        #Masks all emitted frequencies (redshifts) by above. Two arrays of the redshift (and shifted frequency) that fall in contamination window
        nu_observable = nu_shifted[nu_obs_mask]
        z_observable = z_space[nu_obs_mask]
    
        linewidth = linewidth_unshifted * (1 + z_observable)
        deltaZ = z_space[1] - z_space[0]
        
        temp_line = meanT_shifted(E_SFR, tau_L[i], N_HII, nu_emitted, linewidth, bandwidth, z_observable) * deltaZ
        
        #deltaX_n = delX(nu_emitted, 1420, z_observable)
        deltaX_n = np.array([quad(delX_new, z_rrl, z_21)[0] for z_rrl, z_21 in zip(z_observable, contam21cm_z(nu_observable))])
    
        contam_ind.append(T_RRL(temp_line, deltaX_n, z_observable))
        nu_stack.append(nu_observable)
    
    if plot_freqContam == True:
        frequency_contamination(np.linspace(nu_band_min, nu_band_max, 1_00), contam_ind, nu_stack)
    if plot_nContam == True:
        n_contamination(np.arange(n_band_min, n_band_max, 1), contam_ind, n_min)
    
    T_totalContam = np.absolute(np.sum(list(itertools.chain.from_iterable(contam_ind))))
    return(T_totalContam)

