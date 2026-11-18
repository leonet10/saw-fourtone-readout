# Modelo de linea de retardo con un eco, validacion cruzada y estudios de sensibilidad
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares, minimize_scalar

T_REF = 30.0


def phase_model(p, temps, f, dT_local=0.0, c=0.5, m=3):
    # p = [tau0 (us), alpha (ppm/C), eps, dt0 (us), theta (rad)]; devuelve fase desenvuelta (rad), forma len(temps) x len(f)
    tau0, al = p[0] * 1e-6, p[1] * 1e-6
    eps, dt0, th = p[2], p[3] * 1e-6, p[4]
    out = []
    for t in np.atleast_1d(temps):
        s = 1 + al * (t - T_REF)
        tm = tau0 * s + c * tau0 * al * dT_local                           # camino directo: cruza el catalizador una vez
        te = tm + dt0 * s + (m - 1) * c * tau0 * al * dT_local             # eco: cruza el catalizador m veces en total
        H = np.exp(-2j * np.pi * f * tm) + eps * np.exp(-2j * np.pi * f * te + 1j * th)
        out.append(np.unwrap(np.angle(H)))
    return np.array(out)


def _line(p, temps, f):
    # linea ideal (sin eco)
    tau0, al = p[0] * 1e-6, p[1] * 1e-6
    return np.array([-2 * np.pi * f * tau0 * (1 + al * (t - T_REF)) for t in temps])


def _resid(p, P_deg, temps, f, echo):
    W = np.deg2rad(P_deg)
    M = phase_model(p, temps, f) if echo else _line(p, temps, f)
    d = W - M
    d = d - np.round((d - d.mean(1, keepdims=True)) / (2 * np.pi)) * 2 * np.pi
    d = d - d.mean(1, keepdims=True)                                       # desfase libre por barrido (cables)
    return np.rad2deg(d).ravel()


def fit(P_deg, temps, f, echo=True):
    # ajuste por minimos cuadrados no lineales con varios puntos de partida; devuelve (rms_grados, parametros)
    starts = ([[0.992, 68.4, e, 2.28, th] for e in (0.2, -0.2) for th in (0, 1.5, 3, 4.5)]
              if echo else [[0.992, 68.4]])
    best = None
    for p0 in starts:
        r = least_squares(_resid, p0, args=(P_deg, temps, f, echo), max_nfev=5000)
        rms = float(np.sqrt(np.mean(r.fun ** 2)))
        if best is None or rms < best[0]:
            best = (rms, r.x)
    return best


def leave_one_out(P_deg, temps, f):
    # error RMS en el barrido excluido, linea ideal frente a linea + eco
    rows = []
    for k in range(len(temps)):
        tr = np.delete(np.arange(len(temps)), k)
        row = [temps[k]]
        for echo in (False, True):
            p = fit(P_deg[tr], temps[tr], f, echo)[1]
            row.append(float(np.sqrt(np.mean(_resid(p, P_deg[[k]], temps[[k]], f, echo) ** 2))))
        rows.append(row)
    return np.array(rows)


def rigid_shift(P_deg, f):
    # desplazamiento en frecuencia (kHz) que mejor superpone cada barrido al primero, y residuo RMS (grados)
    F = f / 1e6
    ref = CubicSpline(F, P_deg[0])
    out = []
    for p in P_deg:
        def cost(df):
            mk = (F - df > F[0]) & (F - df < F[-1])
            x = p[mk] - ref(F[mk] - df)
            c = np.mean(x - 360 * np.round((x - x.mean()) / 360))
            x = x - 360 * np.round((x - c) / 360)
            return np.mean((x - x.mean()) ** 2)
        r = minimize_scalar(cost, bounds=(-0.4, 0.05), method="bounded", options={"xatol": 1e-7})
        out.append((r.x * 1e3, np.sqrt(r.fun)))
    return np.array(out)


def sensitivity_map(P_deg, f, k_mhz_per_c, f0s, T0s):
    # sensibilidad local d(phi)/dT (grados/C) por desplazamiento en bloque de la curva medida a 30 C
    ref = CubicSpline(f / 1e6, P_deg[0])
    return np.array([[(ref(f0 + k_mhz_per_c * (t + 0.05 - T_REF)) - ref(f0 + k_mhz_per_c * (t - T_REF))) / 0.05
                      for f0 in f0s] for t in T0s])


def tones(fc, K, period_hz):
    # K tonos equiespaciados sobre un periodo de la ondulacion, centrados en fc
    return fc + (np.arange(K) - (K - 1) / 2) * period_hz / K if K > 1 else np.array([fc])


def measured_tone_spread(P_aligned, temps, f, K, period_hz, centers_hz):
    # variacion (max/min) de la sensibilidad entre tramos de 5 C, usando la media de K tonos medidos
    sp = [CubicSpline(f, q) for q in P_aligned]
    v = []
    for fc in centers_hz:
        fr = tones(fc, K, period_hz)
        ph = np.array([s(fr).mean() for s in sp])
        S = np.abs(np.diff(ph) / np.diff(temps))
        v.append(S.max() / S.min())
    return np.array(v)


def local_heating_gain_variation(p, f_centers, T0s, K, m, c=0.5, period_hz=None, slope_rule=False, d=15e3):
    # variacion de ganancia (max/min sobre T0, mediana sobre frecuencias) con calentamiento local del catalizador
    wr = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
    def ph(fr, t0, dT):
        return np.angle(np.exp(1j * phase_model(p, [t0], np.atleast_1d(fr), dT, c, m)[0]))
    var = []
    for fc in f_centers:
        fr = tones(fc, K, period_hz) if K > 1 else np.array([fc])
        S = []
        for t0 in T0s:
            s_true = np.mean(wr(ph(fr, t0, 0.05) - ph(fr, t0, 0.0))) / 0.05
            if slope_rule:                                                   # regla de pendiente con dos tonos
                g = wr(ph(fc + d, t0, 0)[0] - ph(fc - d, t0, 0)[0]) / (2 * d)
                s_true = s_true / (c * fc * p[1] * 1e-6 * g)
            S.append(s_true)
        S = np.abs(S)
        var.append(S.max() / S.min())
    return float(np.median(var))
