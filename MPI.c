#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <string.h>

#define N 4000  // Tamanho da grade
#define T 500   // Número de iterações no tempo
#define D 0.1   // Coeficiente de difusão
#define DELTA_T 0.01
#define DELTA_X 1.0

void diff_eq(double *C, double *C_new, int start, int end, int rank, int size) {
    int i, j;
    MPI_Status status;

    for (int t = 0; t < T; t++) {
        if (rank > 0) {
            MPI_Sendrecv(&C[start * N], N, MPI_DOUBLE, rank - 1, 0,
                         &C[(start - 1) * N], N, MPI_DOUBLE, rank - 1, 0,
                         MPI_COMM_WORLD, &status);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(&C[(end - 1) * N], N, MPI_DOUBLE, rank + 1, 0,
                         &C[end * N], N, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, &status);
        }

        omp_set_num_threads(14);
        #pragma omp parallel for private(j)
        for (i = start; i < end; i++) {
            for (j = 1; j < N - 1; j++) {
                C_new[i * N + j] = C[i * N + j] + D * DELTA_T * (
                    (C[(i+1) * N + j] + C[(i-1) * N + j] + C[i * N + (j+1)] + C[i * N + (j-1)] - 4 * C[i * N + j]) / (DELTA_X * DELTA_X)
                );
            }
        }

        double difmedio = 0.0;
        #pragma omp parallel for private(j) reduction(+: difmedio)
        for (i = start; i < end; i++) {
            for (j = 1; j < N - 1; j++) {
                difmedio += fabs(C_new[i * N + j] - C[i * N + j]);
                C[i * N + j] = C_new[i * N + j];
            }
        }

        double total_difmedio;
        MPI_Reduce(&difmedio, &total_difmedio, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0 && (t % 100) == 0)
            printf("Iteração %d - Diferença Média = %g\n", t, total_difmedio / ((N - 2) * (N - 2)));
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = N / size;
    int start = rank * rows_per_proc;
    int end = (rank + 1) * rows_per_proc;
    if (rank == 0) start = 1;
    if (rank == size - 1) end = N - 1;

    double *C = (double *)malloc(N * N * sizeof(double));
    double *C_new = (double *)malloc(N * N * sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < N * N; i++) {
        C[i] = 0.0;
        C_new[i] = 0.0;
    }

    if (rank == 0) {
        C[(N/2) * N + (N/2)] = 1.0;
    }
    
    MPI_Bcast(&C[(N/2) * N + (N/2)], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    diff_eq(C, C_new, start, end, rank, size);
    double end_time = MPI_Wtime();

    double *C_final = NULL;
    if (rank == 0) {
        C_final = (double *)malloc(N * N * sizeof(double));
        memcpy(C_final, C, (end - start) * N * sizeof(double));
    }

    for (int src = 1; src < size; src++) {
        int src_start = src * (N / size);
        int src_end = (src + 1) * (N / size);
        if (src == 0) src_start = 1;
        if (src == size - 1) src_end = N - 1;

        if (rank == 0) {
            MPI_Recv(&C_final[src_start * N], (src_end - src_start) * N, MPI_DOUBLE, 
                     src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (rank == src) {
            MPI_Send(&C[start * N], (end - start) * N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        printf("Concentração final no centro: %f\n", C_final[(N/2) * N + (N/2)]);
        printf("Tempo total: %f segundos.\n", end_time - start_time);
        free(C_final);
    }

    free(C);
    free(C_new);

    MPI_Finalize();
    return 0;
}
