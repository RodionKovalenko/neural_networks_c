/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.c to edit this template
 */
#include <stdio.h>
#include <stdlib.h>

double** build_array(int n_row, int n_col) {
    int i;
    double **result_matrix = (double**) malloc(n_row * sizeof(double));

    for (i = 0; i < n_row; i++) {
        result_matrix[i] = (double*) malloc(n_col * sizeof(double));
    }

    return result_matrix;
}

double*** build_array_3d(int n_records, int n_row, int n_col) {
    int i,k;
    double ***result_matrix = (double***) malloc(n_records * sizeof (double));

    for (i = 0; i < n_row; i++) {
        result_matrix[i] = (double **) malloc(n_row * sizeof (double));
        for (k = 0; k < n_col; k++) {
            result_matrix[i][k] = (double*) malloc(n_col * sizeof (double));
        }
    }

    return result_matrix;
}

double** clear_array(double **array, int n_row, int n_col) {
    int i, j;

    for (i = 0; i < n_row; i++) {
        for (j = 0; j < n_col; j++) {
            array[i][j] = 0.0;
        }
    }

    return array;
}

/**
 * converts vector to a matrix of (size of Vector, n_col)
 * @param vector
 * @param n_row
 * @param n_col
 * @return 
 */
double** convert_vector_to_matrix(double *vector, int v_dim, int n_col) {
    double **result_matrix = build_array(v_dim, n_col);
    int i, j;

    for (i = 0; i < v_dim; i++) {
        for (j = 0; j < n_col; j++) {
            result_matrix[i][j] = vector[i];
        }
    }

    return result_matrix;
}
