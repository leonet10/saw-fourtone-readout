# Entrenamiento de la red y comparacion de los seis metodos sobre sensores virtuales nuevos
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from .chain import draw_units, measure

wr = lambda x: (x + 180) % 360 - 180


def feats(vb, ve):
    # entradas de la red: voltajes en reposo y diferencias exposicion - reposo (escaladas por 20)
    return np.hstack([vb, (ve - vb) * 20])


def train(chain, fr, hidden, N=150000, seed=1, noise_v=0.0015):
    # aumento de datos con base fisica: solo datos sinteticos del modelo calibrado con mediciones
    r = np.random.default_rng(seed)
    u = draw_units(N, r)
    T0 = r.uniform(20, 60, N)
    dT = r.uniform(0, 6, N)
    X = feats(measure(chain, fr, T0, np.zeros(N), u, r, noise_v), measure(chain, fr, T0, dT, u, r, noise_v))
    sc = StandardScaler().fit(X)
    m = MLPRegressor(hidden_layer_sizes=hidden, activation="tanh", max_iter=400, early_stopping=True,
                     random_state=0, learning_rate_init=2e-3, batch_size=256, n_iter_no_change=20)
    m.fit(sc.transform(X), dT)
    return sc, m


def compare(chain, models, noise_v, N=6000, seed=7):
    # devuelve estimaciones de dT por metodo, dT real, T0 y unidades
    r = np.random.default_rng(seed)
    u = draw_units(N, r)
    T0 = r.uniform(20, 60, N)
    dT = r.uniform(0.2, 6, N)
    z = np.zeros(N)
    out = {}

    def cal(fr, ideal, fn):
        # calibracion de fabrica por sensor: T0 = 25/40/55 C, dT = 2 C, 16 lecturas promediadas por punto
        g = []
        for tc in (25.0, 40.0, 55.0):
            t = np.full(N, tc)
            cb = np.mean([measure(chain, fr, t, z, u, r, noise_v, ideal) for _ in range(16)], 0)
            ce = np.mean([measure(chain, fr, t, np.full(N, 2.0), u, r, noise_v, ideal) for _ in range(16)], 0)
            g.append(fn(cb, ce) / 2.0)
        return np.mean(g, 0)

    for K in (1, 4):
        fr = chain.tones(K)
        vbI = measure(chain, fr, T0, z, u, r, noise_v, True)
        veI = measure(chain, fr, T0, dT, u, r, noise_v, True)
        f_ideal = lambda b, e: wr(e - b).mean(1)
        out[f"{K}T ideal"] = f_ideal(vbI, veI) / cal(fr, True, f_ideal)
        vb = measure(chain, fr, T0, z, u, r, noise_v)
        ve = measure(chain, fr, T0, dT, u, r, noise_v)
        f_naive = lambda b, e: (e - b).mean(1)
        out[f"{K}T AD8302"] = f_naive(vb, ve) / cal(fr, False, f_naive)
        sc, m = models[K]
        f_net = lambda b, e: m.predict(sc.transform(feats(b, e)))
        out[f"{K}T AD8302+NN"] = f_net(vb, ve) / cal(fr, False, f_net)
    return out, dT, T0, u


def summary(out, dT, threshold=1.0):
    # error relativo (%) mediana y percentil 95 para dT >= threshold
    b = dT >= threshold
    return {k: (float(np.nanmedian(100 * np.abs(v[b] - dT[b]) / dT[b])),
                float(np.nanpercentile(100 * np.abs(v[b] - dT[b]) / dT[b], 95))) for k, v in out.items()}


def lod(chain, models, noise_v=0.0015, units=400, repeats=60, seed=11):
    # limite de deteccion 3 sigma (C) por sensor: pares reposo/exposicion repetidos con dT = 0
    r = np.random.default_rng(seed)
    u0 = draw_units(units, r)
    T0u = r.uniform(20, 60, units)
    u = {k: np.repeat(v, repeats, axis=0) for k, v in u0.items()}
    T0 = np.repeat(T0u, repeats)
    z = np.zeros(units * repeats)
    sd = lambda e: e.reshape(units, repeats).std(1)
    res = {}
    for K in (1, 4):
        fr = chain.tones(K)
        pb = measure(chain, fr, T0, z, u, r, noise_v, True)
        pe = measure(chain, fr, T0, z, u, r, noise_v, True)
        g = wr(chain.saw_phase(fr, T0u, np.full(units, 0.05), u0["m"], u0["off"])
               - chain.saw_phase(fr, T0u, np.zeros(units), u0["m"], u0["off"]))
        g = np.abs(g.mean(1)) / 0.05                                       # ganancia real de cada sensor (grados/C)
        res[f"{K}T ideal"] = 3 * sd(wr(pe - pb).mean(1)) / g
    sc, m = models[4]
    fr = chain.tones(4)
    vb = measure(chain, fr, T0, z, u, r, noise_v)
    ve = measure(chain, fr, T0, z, u, r, noise_v)
    res["4T AD8302+NN"] = 3 * sd(m.predict(sc.transform(feats(vb, ve))))
    return res
