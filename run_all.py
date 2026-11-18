# Reproduce todos los resultados del articulo
# Uso: python run_all.py [--data data/]   (sin los CSV medidos, usa model.json y omite las partes medidas)
import argparse
import glob
import json
import os
import subprocess
import numpy as np
from saw_readout import data, model, chain as ch, evaluate as ev, figures, export_c

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="results")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
R = {}

# 1) Datos medidos: ajuste del modelo, validacion cruzada, desplazamiento en bloque, mapa de sensibilidad
if glob.glob(os.path.join(a.data, "*rados*.[Cc][Ss][Vv]")):
    f, T, P = data.load_sweeps(a.data)
    Pa = data.align_branches(P)
    rms_line, _ = model.fit(P, T, f, echo=False)
    rms_echo, p = model.fit(P, T, f, echo=True)
    loto = model.leave_one_out(P, T, f)
    sh = model.rigid_shift(P, f)
    slope = np.polyfit(T, sh[:, 0], 1)
    res_lin = sh[:, 0] - np.polyval(slope, T)
    period = 1 / (p[3] * 1e-6)
    k = abs(slope[0]) / 1e3                                              # MHz/C
    f0s, T0s = np.arange(97.50, 98.501, 0.01), np.arange(25, 60.01, 0.5)
    S = model.sensitivity_map(P, f, -k, f0s, T0s)
    centers = np.arange(97.55, 98.46, 0.03) * 1e6
    spread = {K: model.measured_tone_spread(Pa, T, f, K, period, centers) for K in (1, 4, 8)}
    R["measured"] = dict(rms_line_deg=rms_line, rms_echo_deg=rms_echo, params=list(map(float, p)),
                         echo_dB=float(20 * np.log10(abs(p[2]))), ripple_period_kHz=float(period / 1e3),
                         loto=loto.tolist(), shift_kHz=sh[:, 0].tolist(), shift_resid_deg=sh[:, 1].tolist(),
                         shift_slope_kHz_per_C=float(slope[0]),
                         shift_nonlinearity_pctFS=float(100 * abs(res_lin).max() / abs(sh[-1, 0] - sh[0, 0])),
                         sensitivity_min_max=[float(abs(S).min()), float(abs(S).max())],
                         tone_spread={K: [float(np.median(v)), float(v.max())] for K, v in spread.items()})
    json.dump({"params": list(map(float, p))}, open(os.path.join(a.out, "model.json"), "w"), indent=1)
    figures.fig1(os.path.join(a.out, "fig1_model.png"), f, T, P, p, loto)
    figures.fig2(os.path.join(a.out, "fig2_sensitivity_map.png"), f0s, T0s, S)
    figures.fig4(os.path.join(a.out, "fig4_measured_averaging.png"), f, T, Pa, period)
else:
    p = np.array(json.load(open("model.json"))["params"])               # parametros publicados
    period = 1 / (p[3] * 1e-6)
    print("sin datos medidos: se usa model.json y se omiten las figuras 1, 2 y 4")

# 2) Calentamiento local del catalizador: variacion de ganancia segun el numero de cruces del eco
fsing = np.arange(97.60, 98.40, 0.02) * 1e6
fcent = np.arange(97.70, 98.31, 0.05) * 1e6
T0r = np.arange(25, 60.1, 2.5)
vals = {"Single tone (fixed f₀)": [model.local_heating_gain_variation(p, fsing, T0r, 1, m) for m in (1, 2, 3)],
        "Two-tone slope rule": [model.local_heating_gain_variation(p, fsing, T0r, 1, m, slope_rule=True) for m in (1, 2, 3)],
        "4-tone ripple-period mean": [model.local_heating_gain_variation(p, fcent, T0r, 4, m, period_hz=period) for m in (1, 2, 3)]}
R["local_heating"] = vals
figures.fig3(os.path.join(a.out, "fig3_local_heating.png"), vals)

# 3) Cadena SAW -> AD8302 -> ESP32: entrenamiento, comparacion de metodos y limite de deteccion
C = ch.Chain(p)
models = {1: ev.train(C, C.tones(1), (16, 16)), 4: ev.train(C, C.tones(4), (32, 32))}
R["methods"] = {}
for nv in (1e-5, 0.0015, 0.005):
    out, dT, T0, u = ev.compare(C, models, nv)
    R["methods"][f"noise_{nv / 0.01:.3f}_deg"] = ev.summary(out, dT)
    if nv == 0.0015:
        figures.fig5(os.path.join(a.out, "fig5_methods.png"), out, dT, T0)
        b = dT >= 1
        bins = np.arange(20, 61, 5)
        R["p95_by_T0_bin"] = {k: [float(np.percentile(100 * np.abs(out[k] - dT)[b & (T0 >= lo) & (T0 < lo + 5)]
                                                     / dT[b & (T0 >= lo) & (T0 < lo + 5)], 95)) for lo in bins[:-1]]
                              for k in ("1T ideal", "4T ideal", "4T AD8302+NN")}
L = ev.lod(C, models)
R["lod_C"] = {k: [float(np.median(v)), float(np.percentile(v, 95))] for k, v in L.items()}

# 4) Exportacion a C y verificacion contra Python
os.makedirs("esp32", exist_ok=True)
sc, m = models[4]
R["embedded"] = export_c.export(sc, m, C.period / 4, ch.FC, "esp32")
r = np.random.default_rng(5)
N = 2000
u = ch.draw_units(N, r)
T0 = r.uniform(20, 60, N)
dT = r.uniform(0, 6, N)
fr = C.tones(4)
vb = ch.measure(C, fr, T0, np.zeros(N), u, r)
ve = ch.measure(C, fr, T0, dT, u, r)
np.savetxt("esp32/test_in.txt", np.hstack([vb, ve]), fmt="%.9e")
np.savetxt("esp32/test_py.txt", m.predict(sc.transform(ev.feats(vb, ve))), fmt="%.9e")
try:
    subprocess.run(["gcc", "-O2", "-std=c99", "-Wall", "-Wextra", "-o", "test_c", "test_main.c", "saw_mlp.c", "-lm"],
                   cwd="esp32", check=True)
    R["embedded"]["c_check"] = subprocess.run(["./test_c"], cwd="esp32", capture_output=True, text=True).stdout.strip()
except (OSError, subprocess.CalledProcessError) as e:
    R["embedded"]["c_check"] = f"no se pudo compilar: {e}"

json.dump(R, open(os.path.join(a.out, "results.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
