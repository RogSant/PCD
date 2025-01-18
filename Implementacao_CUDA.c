#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define N 4000 // Tamanho da grade
#define T 500  // Número de iterações no tempo
#define D 0.1  // Coeficiente de difusão
#define DELTA_T 0.01
#define DELTA_X 1.0

__global__ void diff_eq_kernel(double *C, double *C_new, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i < n - 1 && j < n - 1) {
        int idx = i * n + j;
        C_new[idx] = C[idx] + D * DELTA_T * (
            (C[(i + 1) * n + j] + C[(i - 1) * n + j] +
             C[i * n + (j + 1)] + C[i * n + (j - 1)] - 4 * C[idx]) / (DELTA_X * DELTA_X)
        );
    }
}

int main() {

    auto start = std::chrono::high_resolution_clock::now();

    double *C = (double *)malloc(N * N * sizeof(double));
    double *C_new = (double *)malloc(N * N * sizeof(double));

    if (C == NULL || C_new == NULL) {
        fprintf(stderr, "Falha na alocação de memória\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0;
            C_new[i * N + j] = 0.0;
        }
    }

    C[N / 2 * N + N / 2] = 1.0;

    double *d_C, *d_C_new;
    cudaMalloc(&d_C, N * N * sizeof(double));
    cudaMalloc(&d_C_new, N * N * sizeof(double));

    if (d_C == NULL || d_C_new == NULL) {
        fprintf(stderr, "Falha na alocação de memória na GPU\n");
        return 1;
    }

    cudaMemcpy(d_C, C, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_new, C_new, N * N * sizeof(double), cudaMemcpyHostToDevice);
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    for (int t = 0; t < T; t++) {
      diff_eq_kernel<<<gridDim, blockDim>>>(d_C, d_C_new, N);

      double *temp = d_C;
      d_C = d_C_new;
      d_C_new = temp;

      cudaDeviceSynchronize();

      if (t % 100 == 0) {
          cudaMemcpy(C, d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost);
          cudaMemcpy(C_new, d_C_new, N * N * sizeof(double), cudaMemcpyDeviceToHost);

          double difmedio = 0.0;
          for (int i = 1; i < N - 1; i++) {
              for (int j = 1; j < N - 1; j++) {
                  difmedio += fabs(C[i * N + j] - C_new[i * N + j]);
              }
          }

          printf("Iteração %d - diferença média = %g\n", t, difmedio / ((N - 2) * (N - 2)));
      }
}

    cudaMemcpy(C, d_C, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    printf("Concentração final no centro: %f\n", C[N / 2 * N + N / 2]);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> totalTime = end - start;
    printf("Tempo total do programa: %.4f segundos\n", totalTime.count());


    free(C);
    free(C_new);
    cudaFree(d_C);
    cudaFree(d_C_new);

    return 0;
}
