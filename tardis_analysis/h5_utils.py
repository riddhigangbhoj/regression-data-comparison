# tardis_analysis/h5_utils.py

import h5py

def load_h5_data(file_path, keys):
    """Load specified datasets from an HDF5 file."""
    data = {}
    with h5py.File(file_path, 'r') as f:
        for key in keys:
            if key in f:
                data[key] = {
                    'wavelength': f[key]['wavelength'][:],
                    'luminosity': f[key]['luminosity'][:]
                }
    return data

# def load_h5_data(file_path):
#     h5_data = {}
#     with h5py.File(file_path, 'r') as f:
#         for key in ['spectrum_integrated', 'spectrum_real_packets', 'spectrum_real_packets_reabsorbed', 'spectrum_virtual_packets']:
#             group = f'simulation/spectrum_solver/{key}'
#             if group in f:
#                 wavelength = f[group]['wavelength/values'][()]
#                 luminosity = f[group]['luminosity/values'][()]
#                 h5_data[key] = {'wavelength': wavelength, 'luminosity': luminosity}
#                 print(f"Loaded {key}: {len(wavelength)} points")
#             else:
#                 print(f"{key} not found in {file_path}")
#     return h5_data