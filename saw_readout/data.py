# Lectura de los barridos de fase medidos (CSV exportados por el analizador Agilent E5061B)
import glob
import os
import numpy as np

TEMPS = (30, 35, 40, 45, 50, 60)  # temperaturas del horno (C)


def load_sweeps(data_dir):
    # devuelve f (Hz), temperaturas (C) y fase desenvuelta por barrido (grados, forma 6 x N)
    phases, f = [], None
    for t in TEMPS:
        hits = sorted(glob.glob(os.path.join(data_dir, f"*{t}_[Gg]rados*.[Cc][Ss][Vv]")))
        if not hits:
            raise FileNotFoundError(f"no se encontro el barrido de {t} C en {data_dir}")
        d = np.genfromtxt(hits[0], delimiter=",", skip_header=3)   # 3 lineas de cabecera del E5061B
        if f is None:
            f = d[:, 0]
        elif not np.allclose(f, d[:, 0]):
            raise ValueError("los barridos no comparten la misma rejilla de frecuencia")
        phases.append(np.rad2deg(np.unwrap(np.deg2rad(d[:, 1]))))
    return f, np.array(TEMPS, float), np.array(phases)


def align_branches(P):
    # alinea la rama de 360 grados de cada barrido con el anterior, para comparar fases entre temperaturas
    P = P.copy()
    for k in range(1, len(P)):
        P[k] += 360 * np.round((P[k - 1] - P[k]) / 360)
    return P
