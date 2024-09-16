# TiSa_analysis_Code
Analysis software for RABBITT-scans acquired with the VMI-spectrometer in the TiSa-laboratory.

Currently the library consists of two parts that can be used standalone or together:

- `vmi_analysis_library.py` uses the PyAbel package to perform Abel inversions and obtain the photoelectron spectrum.
It can then further be analyzed for oscillations using cosine-fit or fourier-tranform.
Functionality is bundeled into a class.
- `hhg_analysis_library.py` allows reading MCP-images from the HHG-spectrometer and calibrate the energy scale.
