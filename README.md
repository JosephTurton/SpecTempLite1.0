# SpecTemp Lite

Fits temperature, to the nearest 5 k, to an OH(309 nm) spectrum by convolving line strengths with a Gaussian and minimising error vs an experimental CSV.

## Requirements
Python 3.9+ with: numpy, pandas, scipy

## Run
python SpectTemptLite1.0.py /path/to/experimental_spectrum



## Example
Eample_OH_Spectrum is there to run a quick test. SpecTemp should predict the temperature to be 1740 K
Notes:
- Script expects a bundled line list in `my_spectral_data.py` (imported as `CSV`). 
- It peak-aligns around 308.9889945 nm and normalises over ~308.75–309.1 nm. :contentReference[oaicite:2]{index=2}
requirements.txt
