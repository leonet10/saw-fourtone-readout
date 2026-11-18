# Exporta la red entrenada a C (sin dependencias) para el ESP32
import numpy as np

C_SOURCE = r'''// saw_mlp.c -- inferencia de la red (sin dependencias, apta para ESP32)
#include <math.h>
#include "saw_mlp.h"
// vb: VPHS en reposo (V) para los 4 tonos; ve: VPHS durante la exposicion (V). Devuelve dT del catalizador (C) sin calibrar
float saw_mlp_dT(const float vb[4], const float ve[4]) {
    float x[SAW_N_IN], h1[SAW_H1], h2[SAW_H2];
    for (int k = 0; k < 4; k++) { x[k] = vb[k]; x[4 + k] = (ve[k] - vb[k]) * 20.0f; }   // mismas entradas que en Python
    for (int i = 0; i < SAW_N_IN; i++) x[i] = (x[i] - SAW_MU[i]) / SAW_SD[i];          // estandarizacion
    for (int j = 0; j < SAW_H1; j++) {                                                 // capa oculta 1
        float s = SAW_B0[j];
        for (int i = 0; i < SAW_N_IN; i++) s += x[i] * SAW_W0[i * SAW_H1 + j];
        h1[j] = tanhf(s);
    }
    for (int j = 0; j < SAW_H2; j++) {                                                 // capa oculta 2
        float s = SAW_B1[j];
        for (int i = 0; i < SAW_H1; i++) s += h1[i] * SAW_W1[i * SAW_H2 + j];
        h2[j] = tanhf(s);
    }
    float y = SAW_B2[0];
    for (int i = 0; i < SAW_H2; i++) y += h2[i] * SAW_W2[i];
    return y;
}
'''


def _arr(name, a):
    return "static const float %s[%d] = {%s};\n" % (name, a.size, ", ".join("%.9ef" % x for x in a.ravel()))


def export(sc, m, tone_spacing_hz, fc_hz, out_dir):
    W = [w.astype(np.float32) for w in m.coefs_]
    B = [b.astype(np.float32) for b in m.intercepts_]
    h = "// saw_mlp.h -- generado por export_c.py; red 8-32-32-1 (tanh)\n#pragma once\n"
    h += "#define SAW_N_IN %d\n#define SAW_H1 %d\n#define SAW_H2 %d\n" % (W[0].shape[0], W[0].shape[1], W[1].shape[1])
    h += "#define SAW_TONE_SPACING_HZ %.1f\n#define SAW_FC_HZ %.1f\n" % (tone_spacing_hz, fc_hz)
    h += _arr("SAW_MU", sc.mean_.astype(np.float32)) + _arr("SAW_SD", sc.scale_.astype(np.float32))
    for i, (w, b) in enumerate(zip(W, B)):
        h += _arr("SAW_W%d" % i, w) + _arr("SAW_B%d" % i, b)                  # W[i] en orden [entrada][salida]
    open(f"{out_dir}/saw_mlp.h", "w").write(h)
    open(f"{out_dir}/saw_mlp.c", "w").write(C_SOURCE)
    n = sum(w.size for w in W) + sum(b.size for b in B)
    return dict(parameters=int(n), mac=int(sum(w.size for w in W)), tanh=int(W[0].shape[1] + W[1].shape[1]))
