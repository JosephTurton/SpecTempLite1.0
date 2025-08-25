import argparse
import io
from dataclasses import dataclass
from typing import Tuple


import numpy as np
import pandas as pd
from scipy.constants import c, h, k

# ----------------------------
# Data Constants
# ----------------------------


@dataclass(frozen=True)
class ModelConfig:
    T_ref: float = 3000.0
    lambda_min: float = 305.0
    lambda_max: float = 311.0
    n_lambda: int = 2000
    norm_band: Tuple[float, float] = (308.75, 309.10)
    align_centre: float = 308.9889944972486

# ----------------------------
# Utilities
# ----------------------------


def wavelengths_grid(cfg: ModelConfig) -> np.ndarray:
    return np.linspace(cfg.lambda_min, cfg.lambda_max, cfg.n_lambda)


def cm_to_joule(Eu_cm: np.ndarray) -> np.ndarray:
    return h * c * Eu_cm * 100.0


def gaussian_line(lambda_axis: np.ndarray, line_centres: np.ndarray, width: float) -> np.ndarray:
    assert lambda_axis.ndim == 1 and line_centres.ndim == 1
    diff = lambda_axis[:, None] - line_centres[None, :]
    return (2 / width * np.sqrt(np.pi)) * np.exp(- ((diff)**2 / (width / 2)**2))


def normalise_to_band(x: np.ndarray, y: np.ndarray, band: Tuple[float, float]):
    mask = (x > band[0]) & (x < band[1])
    peak = np.max(y[mask]) if np.any(mask) else np.max(y)
    return y / peak


def peak_align(x: np.ndarray, y: np.ndarray, target_centre: float):
    i_peak = int(np.argmax(y))
    gap = target_centre - x[i_peak]
    return x + gap, y

# ----------------------------
# Model
# ----------------------------


class SpectralModel:
    def __init__(self, wavelengths_nm: np.ndarray, I_ref: np.ndarray, Eu_cm: np.ndarray, cfg: ModelConfig):
        self.cfg = cfg
        self.line_centres = wavelengths_nm.astype(float)
        self.I_ref = I_ref.astype(float)
        self.Eu_J = cm_to_joule(Eu_cm.astype(float))
        self.lambda_axis = wavelengths_grid(cfg)
        self.norm_mask = (self.lambda_axis > cfg.norm_band[0]) & (
            self.lambda_axis < cfg.norm_band[1])

    def line_intensities(self, T: float) -> np.ndarray:
        T_ref = self.cfg.T_ref
        expo = -self.Eu_J * (T_ref - T) / (T_ref * T * k)
        I_T = self.I_ref * np.exp(expo)
        return I_T

    def synthesise(self, T: float, width: float) -> np.ndarray:
        I_T = self.line_intensities(T)
        G = gaussian_line(self.lambda_axis, self.line_centres, width)
        spectrum = G @ I_T
        return normalise_to_band(self.lambda_axis, spectrum, self.cfg.norm_band)

# -----------------------
# Fitting
# -----------------------


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def coarse_search(model: SpectralModel, exp_resampled: np.ndarray, T_grid: np.ndarray,
                  Delta_grid: np.ndarray) -> Tuple[float, float, float]:
    best_err = np.inf
    best_T = float(T_grid[0])
    best_D = float(Delta_grid[0])

    for D in Delta_grid:
        specs = np.stack([model.synthesise(T, D) for T in T_grid], axis=0)
        errs = np.mean((specs - exp_resampled[None, :]) ** 2, axis=1)
        i = int(np.argmin(errs))
        if errs[i] < best_err:
            best_err = float(errs[i])
            best_T = float(T_grid[i])
            best_D = float(D)
    return best_T, best_D, best_err


def refine_T(model: SpectralModel, exp_resampled: np.ndarray, T_centre: float, window: int = 200, step: int = 5, width: float = 0.05) -> Tuple[float, float]:
    T_grid = np.arange(max(100.0, T_centre - window),
                       T_centre + window + step, step)
    specs = np.stack([model.synthesise(T, width) for T in T_grid], axis=0)
    errs = np.mean((specs - exp_resampled[None, :]) ** 2, axis=1)
    i = int(np.argmin(errs))
    return float(T_grid[i]), float(errs[i])

# ------------------------
# I/O
# ------------------------


def load_reference(data_module: str = "my_spectral_data") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mod = __import__(data_module, fromlist=["CSV"])
    df = pd.read_csv(io.StringIO(mod.CSV))
    return df["Wavelength"].to_numpy(), df["I_ref (3000 K)"].to_numpy(), df["E_u (cm⁻¹)"].to_numpy()


def load_experiment(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    x = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float)
    y = y - np.min(y)
    return x, y


def resample_to_axis(x: np.ndarray, y: np.ndarray, x_axis: np.ndarray) -> np.ndarray:
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]
    return np.interp(x_axis, x_sorted, y_sorted, left=0.0, right=0.0)

# ------------------------
# Main
# ------------------------


def run(csv_path: str, data_module: str = "my_spectral_data") -> None:
    cfg = ModelConfig()

    wl_nm, I_ref, Eu_cm = load_reference(data_module)
    model = SpectralModel(wl_nm, I_ref, Eu_cm, cfg)

    x_raw, y_raw = load_experiment(csv_path)
    x_aligned, y_aligned = peak_align(x_raw, y_raw, cfg.align_centre)

    exp_resampled = resample_to_axis(x_aligned, y_aligned, model.lambda_axis)
    exp_resampled = normalise_to_band(model.lambda_axis,
                                      exp_resampled, cfg.norm_band)

    # Course Grid
    T_grid = np.arange(200.0, 3200.0, 200.0)
    Delta_grid = np.arange(0.02, 0.11, 0.01)

    best_T, best_D, _ = coarse_search(model, exp_resampled, T_grid, Delta_grid)
    T_refined, _ = refine_T(model, exp_resampled,
                            T_centre=best_T, window=200, step=5, width=best_D)

    print(f"Estimated Temperature: {T_refined:.0f} K (FWHM ~ {best_D:.3f} nm)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate temperature from OH-309 spectrum.")
    parser.add_argument(
        "csv_path", help="Path to experimental spectrum CSV (two columns: wavelength[nm], intensity)")
    parser.add_argument("--data-module", default="my_spectral_data",
                        help="Module name that contains the reference CSV string 'CSV'")
    args = parser.parse_args()
    run(args.csv_path, args.data_module)
