# Measured data (not included)

Available from the author upon reasonable request (lperez@universidadean.edu.co).

`run_all.py --data data/` expects six CSV files exported by an Agilent E5061B network analyzer, one per oven
temperature, named `30_Grados.CSV`, `35_Grados.CSV`, `40_Grados.CSV`, `45_Grados.CSV`, `50_Grados.CSV` and
`60_Grados.CSV` (any prefix is accepted). Each file has three header lines followed by rows of
`frequency (Hz), phase (degrees), 0`, with the same 201-point frequency grid in all files.
