// Verifica la inferencia en C contra los valores de referencia de Python
#include <stdio.h>
#include <math.h>
float saw_mlp_dT(const float vb[4], const float ve[4]);
int main(void) {
    FILE *fi = fopen("test_in.txt", "r"), *fp = fopen("test_py.txt", "r");
    if (!fi || !fp) { printf("faltan test_in.txt o test_py.txt\n"); return 1; }
    float vb[4], ve[4]; double yp, maxd = 0; int n = 0;
    while (fscanf(fi, "%f %f %f %f %f %f %f %f", &vb[0], &vb[1], &vb[2], &vb[3], &ve[0], &ve[1], &ve[2], &ve[3]) == 8
           && fscanf(fp, "%lf", &yp) == 1) {
        double d = fabs(saw_mlp_dT(vb, ve) - yp);
        if (d > maxd) maxd = d;
        n++;
    }
    printf("vectores %d, diferencia maxima C vs Python: %.2e C\n", n, maxd);
    return 0;
}
