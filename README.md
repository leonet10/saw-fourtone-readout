# Ripple-period four-tone phase readout for a catalytic SAW hydrogen sensor

Code for the article *"Ripple-period multi-tone phase readout with embedded machine-learning correction for a
catalytic PdPt/Al₂O₃ SAW hydrogen sensor"* (L. Perez-Cortes, Universidad EAN).

It reproduces every number and figure of the article:

| Step | What it does | Needs measured data |
| --- | --- | --- |
| 1 | Fits the delay-line + echo model, leave-one-temperature-out validation, rigid frequency shift, sensitivity map, four-tone validation (Figs. 1, 2, 4) | Yes |
| 2 | Local catalyst heating: gain variation for a single tone, the two-tone slope rule and the four-tone mean (Fig. 3) | No (uses `model.json`) |
| 3 | Monte Carlo of the SAW → AD8302 → ESP32 chain, network training, comparison of six methods, limit of detection (Fig. 5) | No |
| 4 | Export of the network to C and check against Python on 2000 vectors | No |

## Run

```bash
pip install -r requirements.txt
python run_all.py --data data/        # with the measured sweeps
python run_all.py                     # without them: steps 2-4 from model.json
```

Results go to `results/` (`results.json` and the figures); the C files go to `esp32/`. A full run takes about
one minute on a laptop.

## Measured data

The phase sweeps (Agilent E5061B, 97.25–98.75 MHz, 201 points, oven at 30–60 °C) were acquired during the
author's doctoral research at Université Grenoble Alpes (Laboratoire SYMME, 2017). They are available from the
author upon reasonable request; see `data/README.md` for the expected file format. `model.json` holds the five
model parameters fitted on these data, so the simulation can be reproduced without them.

## Embedded network

`esp32/saw_mlp.h` and `esp32/saw_mlp.c` implement the 8-32-32-1 tanh network in plain C99 (1377 parameters,
1312 multiply–accumulate operations, 64 `tanhf` per inference). Inputs: the four AD8302 `VPHS` voltages at
baseline and during exposure. Output: the catalyst temperature rise ΔT before the per-sensor gain calibration.

```c
float dT = saw_mlp_dT(vb, ve) / gain;   // gain from the factory calibration of each sensor
```

## Repository layout

```
run_all.py              reproduces all results
model.json              fitted model parameters (tau0 us, alpha ppm/C, eps, dt0 us, theta rad)
saw_readout/data.py     reading of the E5061B CSV sweeps
saw_readout/model.py    delay-line + echo model, fit, validation, sensitivity studies
saw_readout/chain.py    SAW, AD8302 and ESP32 ADC models
saw_readout/evaluate.py network training, comparison of methods, limit of detection
saw_readout/figures.py  figures of the article
saw_readout/export_c.py export of the network to C
esp32/                  C inference and its test harness
```

## Citation

Please cite the article (reference to be added on publication) and, for the measured data, L. A. Pérez-Cortés,
*Conception et réalisation d'un capteur SAW à base de matériaux catalytiques nanostructurés pour la détection du
monoxyde de carbone et de l'hydrogène*, Ph.D. thesis, Université Grenoble Alpes, 2017.

## AI assistance

Parts of this code were developed with the assistance of Claude (Anthropic). All code was reviewed and run by the
author, and the C inference is checked against Python on every run.

## License

MIT, see `LICENSE`.
