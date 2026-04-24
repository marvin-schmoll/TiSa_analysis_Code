# TiSa_analysis_Code
Analysis software for RABBITT-scans acquired with the VMI-spectrometer in the TiSa-laboratory.

Currently the library consists of two parts that can be used standalone or together:

- `vmi_analysis_library.py` uses the PyAbel package to calculate Abel inversions with the rbasex transform and obtain the photoelectron spectrum.
It can then further be analyzed for oscillations using cosine-fit or fourier-tranform.
Functionality is bundeled into two_classes, one dealing with the VMI images directly and one responsible for the RABBITT analysis.
- `hhg_analysis_library.py` allows reading MCP-images from the HHG-spectrometer and calibrate the energy scale.

Independent helper functions needed for one or both files can be found in `utility_library.py` 
