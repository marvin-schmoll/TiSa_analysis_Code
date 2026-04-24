# TiSa_analysis_Code
Analysis software for RABBITT-scans acquired with the VMI-spectrometer in the TiSa-laboratory.
Supports Abel inversion with the PyAbel package, angular integration as well as standard fitting/fourier transform phase analysis techniques.

Currently the library consists of two parts that can be used standalone or together:

- `vmi_analysis_library.py` uses the PyAbel package to calculate Abel inversions with the rbasex transform and obtain the photoelectron spectrum.
It can then further be analyzed for oscillations using cosine-fit or fourier-tranform.
Functionality is bundeled into two classes, one dealing with the VMI images directly and one responsible for the RABBITT analysis.
Reading of spectra directly into the RABBITT-class is now also supported as long as an energy axis is provided.
- `hhg_analysis_library.py` allows reading MCP-images from the HHG-spectrometer and calibrate the energy scale.

Independent helper functions needed for one or both files can be found in `utility_library.py` 
