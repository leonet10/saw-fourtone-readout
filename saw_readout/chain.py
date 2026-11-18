# Cadena de lectura simulada: SAW (linea + eco, calentamiento local) -> AD8302 -> ADC del ESP32
import numpy as np

FC = 98.0e6  # frecuencia central de los tonos (Hz)


class Chain:
    def __init__(self, p, c=0.5):
        # p = parametros ajustados del modelo [tau0 us, alpha ppm/C, eps, dt0 us, theta rad]
        self.tau0, self.al = p[0] * 1e-6, p[1] * 1e-6
        self.eps, self.dt0, self.th = p[2], p[3] * 1e-6, p[4]
        self.c = c
        self.period = 1 / self.dt0                                          # periodo de la ondulacion (Hz)

    def tones(self, K, fc=FC):
        return fc + (np.arange(K) - (K - 1) / 2) * self.period / K if K > 1 else np.array([fc])

    def saw_phase(self, fr, T0, dT, m, off):
        # fase envuelta (grados), forma N x K; T0, dT, m, off: vectores de longitud N
        s = 1 + self.al * (T0[:, None] - 30)
        tm = self.tau0 * s + self.c * self.tau0 * self.al * dT[:, None]
        te = tm + self.dt0 * s + (m[:, None] - 1) * self.c * self.tau0 * self.al * dT[:, None]
        H = np.exp(-2j * np.pi * fr * tm) + self.eps * np.exp(-2j * np.pi * fr * te + 1j * self.th)
        return np.rad2deg(np.angle(H * np.exp(1j * np.deg2rad(off[:, None]))))


def ad8302(phi, slope, icpt, corner=6.0):
    # |phi| (0-180 grados) -> VPHS (V): 10 mV/grado nominal, 1.8 V a 0 grados, compresion suave cerca de 0 y 180
    a = np.abs((phi + 180) % 360 - 180)
    a = a + 3 * np.exp(-a / corner) - 3 * np.exp(-(180 - a) / corner)
    return icpt - slope * a


def esp32_adc(v, inl, noise, rng):
    # recorte al rango util 0.15-2.45 V (atenuacion 11 dB), INL cubica por unidad, ruido tras promediar, 12 bits
    vc = np.clip(v, 0.15, 2.45)
    x = (vc - 1.3) / 1.15
    err = inl[:, 0:1] * x ** 3 + inl[:, 1:2] * x ** 2 + inl[:, 2:3] * (x ** 3 - x)
    code = np.round((vc + err + rng.normal(0, noise, v.shape)) / 2.45 * 4095)
    return code * 2.45 / 4095


def draw_units(N, rng, m_set=(1, 2, 3)):
    # parametros aleatorios de cada sensor virtual
    return dict(m=rng.choice(m_set, N).astype(float),
                off=rng.uniform(-180, 180, N),            # fase de cables y conectores (grados)
                slope=rng.normal(0.010, 0.0003, N),       # pendiente del AD8302 (V/grado), +-3 %
                icpt=rng.normal(1.80, 0.02, N),           # intercepto del AD8302 (V)
                inl=rng.normal(0, 0.008, (N, 3)))         # coeficientes de INL del ADC (V)


def measure(chain, fr, T0, dT, u, rng, noise_v=0.0015, ideal=False):
    # una lectura de los K tonos: fase con signo (detector ideal, grados) o voltaje del ADC (V)
    phi = chain.saw_phase(fr, T0, dT, u["m"], u["off"])
    if ideal:
        return phi + rng.normal(0, noise_v / 0.010, phi.shape)
    v = ad8302(phi, u["slope"][:, None], u["icpt"][:, None])
    return esp32_adc(v, u["inl"], noise_v, rng)
