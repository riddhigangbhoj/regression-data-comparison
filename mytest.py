import tardis
import numpy as np
from tardis import run_tardis

# Define the configuration file path
# You'll need to create or have a YAML configuration file for TARDIS
config_file = "path/to/your/tardis_config.yml"

# Run TARDIS simulation
simulation = run_tardis(config_file)

# Access the simulation results
spectrum = simulation.spectrum
wavelength = spectrum.wavelength
luminosity = spectrum.luminosity

# You can plot the results using matplotlib
import matplotlib.pyplot as plt

plt.figure()
plt.plot(wavelength, luminosity)
plt.xlabel('Wavelength (Å)')
plt.ylabel('Luminosity')
plt.title('TARDIS Spectrum')
plt.show()
