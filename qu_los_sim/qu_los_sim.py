#!/usr/bin/env python3

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from RMtools_1D.do_RMsynth_1D import run_rmsynth
from RMtools_1D.do_RMclean_1D import run_rmclean
from RMutils.util_RM import get_rmsf_planes


def make_los(parameters : Dict,  
             freqs      : Optional[Union[List[float], np.ndarray]] = None,
             noise      : float = 0.001,
             output_file: str = None,
             do_rmsynth : bool = True) -> Dict[str, np.ndarray]:
    """
    Calculate the complex polarisation for a set of input models along the LOS.
    Optionally performs RM synthesis on the resulting complex polarisation.
    
    Parameters:
    -----------
    parameters : dict
        Dictionary containing lists of model parameters with the following 
        required keys:
        - 'pi'   : list of float, polarised intensity/intensities (any units)
        - 'phi'  : list of float, Faraday depth(s) (rad/m^2)
        - 'psi'  : list of float, initial polarisation angle(s) (degrees)
        - 'sig'  : list of float, depolarisation parameter(s)
        - 'dphi' : list of float, Faraday depth width(s) for Burn slab(s)
        All lists must have the same length.
    
    freqs : array-like, optional
        Array containing observing frequencies (in Hz).
        Default is 300-1800 MHz with 1 MHz spacing.
    
    noise : float, optional
        Standard deviation of Gaussian noise to include in Q and U
        (in same units as polarised intensity). Default is 0.001.
    
    output_file : str, optional
        Name of output file to save the results. If not provided,
        results are returned but not saved to a file.

    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'pol' : complex ndarray, complex polarisation (Q + iU)
        - 'freq': ndarray, frequency array (Hz)
        - 'lsq' : ndarray, wavelength squared (m^2)
        - 'fd'  : ndarray, Faraday depth spectrum
        - 'fdf_dirty' : ndarray, dirty Faraday dispersion function


    Examples:
    --------
    >>> parameters = {
            "pi"  : [1.0, 0.5],
            "phi" : [10.0, -5.0],
            "psi" : [0.0, 45.0],
            "sig" : [0.0, 0.0],
            "dphi": [0.0, 0.0]
        }
    >>> results = make_los(parameters)
    
    Notes:
    ------
    The code supports three types of models:
    1. Faraday screen without depolarisation (sig=0, dphi=0)
    2. Faraday screen with Burn depolarisation (sig≠0, dphi=0)
    3. Burn slab (dphi≠0)
    The RM synthesis option requires CIRADA RM Tools to be installed on 
    the system: https://github.com/CIRADA-Tools/RM-Tools


    Development Notes:
    ----------------
    Authors: Anna Ordog, Alex Hill
    Created: February 2025
    Code development was assisted by Anthropic's Claude 3.5 Sonnet (2024),
    particularly for parameter validation, error handling, documentation
    structure.
    """

    # Frequency array (read in or generate default)
    if freqs is None:
        freqs = np.arange(300e6,1801e6,1e6)
    else:
        freqs = np.array(freqs)
    lsq = ((3e8)/freqs)**2

    # Validate parameters
    check_parameters(parameters)

    # Make the complex LOS:
    complex_pol = sim_qu(parameters, lsq)

    # Add noise:
    q_noise = np.random.normal(loc=np.zeros_like(freqs), scale=noise*np.ones_like(freqs))
    u_noise = np.random.normal(loc=np.zeros_like(freqs), scale=noise*np.ones_like(freqs))
    complex_pol.real = complex_pol.real+q_noise
    complex_pol.imag = complex_pol.imag+u_noise

    # Assemble results into dictionary:
    results = {'pol' :complex_pol,
               'freq':freqs,
               'lsq' :lsq}

    # RM synthesis:
    if do_rmsynth:
        rmsfplanes_out = get_rmsf_planes(lsq, np.arange(-200,200,0.1))
        rmsynth_out    = run_rmsynth([freqs, complex_pol.real, 
                                             complex_pol.imag,
                                             np.ones_like(complex_pol.real)*noise, 
                                             np.ones_like(complex_pol.imag)*noise],
                                             dPhi_radm2=0.1, phiMax_radm2=200)
        # Include results in dictionary:
        results['fd'] = rmsfplanes_out
        results['fdf_dirty'] = rmsynth_out
    
        
    # If output file is specified, save results
    if output_file:
        with open(output_file, 'w') as f:
            f.write("# Line of sight calculations\n")
            for model_name, data in results.items():
                f.write(f"\n# Model: {model_name}\n")
                np.savetxt(f, data)
    
    return results


def check_parameters(parameters: Dict) -> None:
    """
    Validate parameter dictionary structure.
    
    Parameters:
    -----------
    parameters : dict
        Dictionary containing model parameters
        
    Raises:
    -------
    ValueError
        If required parameters are missing or lists have inconsistent lengths
    """
    # Required parameters
    required_params = {"pi", "phi", "psi", "sig", "dphi"}
    
    # Check if all required parameters exist
    missing_params = required_params - set(parameters.keys())
    if missing_params:
        raise ValueError(f"Missing required parameters: {missing_params}")
    
    # Get length of first parameter's list
    first_param = list(required_params)[0]
    required_length = len(parameters[first_param])
    
    # Check if at least one model is included
    if required_length < 1:
        raise ValueError("Parameter lists must contain at least one element")
    
    # Check if all required parameters have the same length
    inconsistent_params = [param for param in required_params 
                         if len(parameters[param]) != required_length]
    if inconsistent_params:
        raise ValueError(f"Parameters {inconsistent_params} have inconsistent length. "
                       f"All parameter lists must have the same length")
    
    # Check for extra parameters
    extra_params = set(parameters.keys()) - required_params
    if extra_params:
        print(f"Warning: Additional parameters found and will be ignored: {extra_params}")

    return



def sim_qu(parameters, lsq):
    """
    Calculate the complex polarisation for the set of input models
    to include along the LOS.
    
    Parameters:
    -----------
    parameters : dict
        Dictionary containing model parameters
    lsq : array
        Array containing wavelength squared values for sampling

    Returns:
    --------
    pol : complex array
        Array of complex values (Stoke Q - real, Stokes U - imag) 

    """
    print('')
    pol = np.zeros_like(lsq)

    # Loop through all models included for the LOS
    for i in range(0,len(parameters["pi"])):

        # Pick out the model
        p_set = {key: values[i] for key, values in parameters.items()}

        # Determine the model type based on the non-zero parameters
        model_type, model_params = get_model_type(p_set)
        print(model_type)
        print(model_params)
        print('')

        # The supported model types/components
        screen = np.exp(2j*(p_set['phi']*lsq + np.radians(p_set['psi'])))
        depol  = np.exp(-2 *p_set['sig']**2 * lsq**2)
        slab   = np.sin(p_set['dphi']*lsq)/(p_set['dphi']*lsq)

        # Build the model (slab or screen)
        if model_type == 'Burn slab':
            pol = pol + p_set['pi']*screen*depol*slab
        else:
            pol = pol + p_set['pi']*screen*depol

    return pol



def get_model_type(param_set):
    """
    Checks type of model based on parameter values and returns the type.
    
    Parameters:
    -----------
    param_set : dict
        Dictionary containing model parameters
        
    Returns:
    -------
    model_type : str
        String describing the model type. Recognized models:
        - Faraday screen, no depolarisation
        - Faraday screen with Burn depolarisation
        - Burn slab
    model_params : str
        String summarizing the parameter values
    """

    # Check for screen model (slab width is zero)
    if param_set["dphi"] == 0.0:
        # Check for depolarisation component
        if param_set["sig"] == 0.0:
            model_type   = 'Faraday screen, no depolarization'
            model_params = (f'PI={param_set["pi"]}, '
                            f'FD={param_set["phi"]}, '
                            f'psi0={param_set["psi"]}')
        else:
            model_type = 'Faraday screen with Burn depolarization'
            model_params = (f'PI={param_set["pi"]}, '
                            f'FD={param_set["phi"]}, '
                            f'psi0={param_set["psi"]}, '
                            f'sigma={param_set["sig"]}')
    # Currently, alternative to screen is slab (slab width non-zero)
    else:
        model_type = 'Burn slab'
        model_params = (f'PI={param_set["pi"]}, '
                        f'FD={param_set["phi"]}, '
                        f'psi0={param_set["psi"]}, '
                        f'sigma={param_set["sig"]}, '
                        f'dFD={param_set["dphi"]}')
        
    return model_type, model_params

