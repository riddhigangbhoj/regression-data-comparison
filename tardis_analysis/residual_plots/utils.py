import numpy as np

def calculate_residuals(wavelength, luminosity, ref_wavelength, ref_luminosity):
    """
    Calculate fractional residuals between two spectra.
    
    Parameters
    ----------
    wavelength : ndarray
        Wavelength array of the spectrum to compare
    luminosity : ndarray
        Luminosity array of the spectrum to compare
    ref_wavelength : ndarray
        Wavelength array of the reference spectrum
    ref_luminosity : ndarray
        Luminosity array of the reference spectrum
        
    Returns
    -------
    wavelength : ndarray
        Wavelength array (if matching reference)
    residuals : ndarray
        Calculated residuals
    is_valid : bool
        Whether the wavelengths match and calculation was successful
    """
    # Ensure wavelengths match the reference
    if not np.array_equal(wavelength, ref_wavelength):
        return None, None, False
        
    # Calculate fractional residuals, handling division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        residuals = np.where(ref_luminosity != 0, 
                             (luminosity - ref_luminosity) / ref_luminosity, 
                             0)
    return wavelength, residuals, True