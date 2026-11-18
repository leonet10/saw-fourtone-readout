// saw_mlp.c -- inferencia de la red (sin dependencias, apta para ESP32)
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
