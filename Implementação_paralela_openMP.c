#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#define N 4000  // Tamanho da grade
#define T 500 // Número de iterações no tempo
#define D 0.1  // Coeficiente de difusão
#define DELTA_T 0.01
#define DELTA_X 1.0

void diff_eq(double **C, double **C_new) { //diff_eq(double C[N][N], double C_new[N][N]) {
int i,j;
    for (int t = 0; t < T; t++) {
      omp_set_num_threads(14);
      #pragma omp parallel for private(j)
        for (int i = 1; i < N - 1; i++) {
          for (j = 1; j < N - 1; j++) {
              C_new[i][j] = C[i][j] + D * DELTA_T * (
                  (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4 * C[i][j]) / (DELTA_X * DELTA_X)
              );
          }
        }
        // Atualizar matriz para a próxima iteração
        double difmedio = 0.;
        #pragma omp parallel for private(j) reduction(+: difmedio)
        for (int i = 1; i < N - 1; i++) {
          for (int j = 1; j < N - 1; j++) {
              difmedio += fabs(C_new[i][j] - C[i][j]);
              C[i][j] = C_new[i][j];
          }
        }
        if ((t%100) == 0)
          printf("interacao %d - diferenca=%g\n", t, difmedio/((N-2)*(N-2)));
    }
}

int main() {

  printf("threads\n");
  long int j, k;
  double start = omp_get_wtime();

    // Concentração inicial
    double **C = (double **)malloc(N * sizeof(double *));
    if (C == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      return 1;
    }
    for (int i = 0; i < N; i++) {
      C[i] = (double *)malloc(N * sizeof(double));
      if (C[i] == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
      }
    }
    #pragma omp parallel for private(j)
    for (int i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
        C[i][j] = 0.;
      }
    }

    // Concentração para a próxima iteração
    double **C_new = (double **)malloc(N * sizeof(double *));
    if (C_new == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      return 1;
    }
    for (int i = 0; i < N; i++) {
      C_new[i] = (double *)malloc(N * sizeof(double));
      if (C_new[i] == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
      }
    }
    #pragma omp parallel for private(k)
    for (int i = 0; i < N; i++) {
      for (k = 0; k < N; k++) {
        C_new[i][j] = 0.;
      }
    }

    // Inicializar uma concentração alta no centro
    C[N/2][N/2] = 1.0;

    // Executar as iterações no tempo para a equação de difusão
    diff_eq(C, C_new);

    // Exibir resultado para verificação
    printf("Concentração final no centro: %f\n", C[N/2][N/2]);
    double end = omp_get_wtime();
    printf(" took %f seconds.\n", end-start);
    return 0;
}
