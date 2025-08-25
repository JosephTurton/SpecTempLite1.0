from my_spectral_data import CSV
import numpy as np
import pandas as pd
import io
from scipy.constants import c, h, k
from scipy.interpolate import interp1d


df = pd.read_csv(io.StringIO(CSV))

# Extract columns as lists
wavelengths = np.array(df["Wavelength"].tolist())
I_ref = np.array(df["I_ref (3000 K)"].tolist())
Eu_cm = np.array(df["E_u (cm⁻¹)"].tolist())


T_ref = 3000
lambda_range = np.linspace(305, 311, 2000)
delta_lambda = 0.01
Inc = 5

## === Functions === ##


def eu_to_J(Eu_cm):
    """Convert energy from cm⁻¹ to Joules."""
    return h * c * Eu_cm * 100


def gaussian_line(x, center, width):
    """Return a Gaussian line profile centered at `center` with FWHM `width`."""
    return (2 / width * np.sqrt(np.pi)) * np.exp(- ((x - center)**2 / (width / 2)**2))


## Creates Dirac Impulses ##
def impulses(T):
    Eu_J = eu_to_J(Eu_cm)
    I_T = I_ref * np.exp(-Eu_J * (T_ref - T) / (T_ref * T * k))
    Gref = I_T[(wavelengths > 308.75) & (wavelengths < 309.1)].max()
    return I_T / Gref


## Generates a convoluded spectrum ##
def Generate(T):
    spectrum = np.zeros_like(lambda_range)
    intensities = impulses(T)
    for wl, I_T in zip(wavelengths, intensities):
        spectrum += I_T * gaussian_line(lambda_range, wl, delta_lambda)
    spectrum /= spectrum[(lambda_range > 308.75) &
                         (lambda_range < 309.1)].max()
    return spectrum


# exp_filepath = "C:/Users/turtl/OH_Sumer_Project/Data/oh-309.csv"
exp_filepath = input('Please enter the spectrum file path...')

df_exp = pd.read_csv(exp_filepath)
Intensity = df_exp.iloc[:, 1].values
lambda_ = df_exp.iloc[:, 0].values
Intensity = Intensity - Intensity.min()

peak_index = np.argmax(Intensity)
gap = 308.9889944972486 - lambda_[peak_index]
if gap > 0:
    lambda_ = lambda_ + gap
else:
    lambda_ = lambda_ - gap
Intensity = Intensity.astype(
    float) / float(Intensity[(lambda_ > 308.5) & (lambda_ < 309.1)].max())

interp_exp = interp1d(lambda_, Intensity, kind='linear',
                      bounds_error=False, fill_value=0)
exp_resampled = interp_exp(lambda_range)

temps = range(600, 3200, 200)
Delta = np.arange(0.02, 0.11, 0.01)
results = []
print('Computing...')
for D in Delta:
    delta_lambda = D
    for T in temps:
        spec = Generate(T)
results.append((spec, D))

errors = [np.mean((s - exp_resampled)**2) for s, _ in results]
best_index = np.argmin(errors)
best_s, best_delta = results[best_index]
print(round(best_delta, 2))

delta_lambda = best_delta
spectra = [Generate(T) for T in temps]
errors = [np.mean((s - exp_resampled)**2) for s in spectra]
best_fit_index = np.argsort(errors)
l, u = sorted([temps[best_fit_index[0]], temps[best_fit_index[1]]])

temps = range(l, u, Inc)
spectra = [Generate(T) for T in temps]

errors = [np.mean((s - exp_resampled)**2) for s in spectra]
best_temperature = temps[np.argmin(errors)]

print(best_temperature)
