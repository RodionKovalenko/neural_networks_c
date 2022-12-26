/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.c to edit this template
 */

#include <stdio.h>
#include <stdlib.h>
#include<time.h>
#include <math.h>
#include <float.h>

double** matrix_product(double **matrix1, double **matrix2, int row, int col, int col_k) {
    int i, j, k;
    double **matrix3;

    matrix3 = malloc(sizeof (double) * row);

    for (i = 0; i < row; i++) {
        matrix3[i] = malloc(sizeof (double*) * col);
    }
    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            for (k = 0; k < col_k; k++) {
                matrix3[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    return matrix3;
}

double get_random_value() {
    double random_value;

    random_value = (double) rand() / RAND_MAX * 2.0 - 1.0; //float in range -1 to 1

    return 0.1;

    //return random_value;
}

double** init_weight_matrix(int weight_r, int weight_c) {
    int i, j;

    double **weight_matrix = (double**) malloc(weight_r * sizeof (double));
    for (i = 0; i < weight_r; i++)
        weight_matrix[i] = (double*) malloc(weight_c * sizeof (double));

    for (i = 0; i < weight_r; i++) {
        for (j = 0; j < weight_c; j++) {
            weight_matrix[i][j] = get_random_value();
        }
    }

    return weight_matrix;
}

double sigmoid_value(double value) {
    return (1.0 / (1.0 + exp(-value)));
}

double** sigmoid_to_matrix(double **matrix, int row, int col) {
    int i, j;

    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            matrix[i][j] = sigmoid_value(matrix[i][j]);
        }
    }

    return matrix;
}

double sigmoid_derivate_to_value(double value) {
    return (value * (1.0 - value));
}

double** matrix_subtract(double **matrix_1, double **matrix_2, int row, int col) {
    int i, j;

    double **result_matrix = (double**) malloc(row * sizeof (double));

    for (i = 0; i < row; i++)
        result_matrix[i] = (double*) malloc(col * sizeof (double));

    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            result_matrix[i][j] = matrix_1[i][j] - matrix_2[i][j];
        }
    }

    return result_matrix;
}

double** matrix_sum(double **matrix_1, double **matrix_2, int row, int col) {
    int i, j;

    double **result_matrix = (double**) malloc(row * sizeof (double));

    for (i = 0; i < row; i++)
        result_matrix[i] = (double*) malloc(col * sizeof (double));

    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            result_matrix[i][j] = matrix_1[i][j] + matrix_2[i][j];
        }
    }

    return result_matrix;
}

double** sigmoid_derivative(double **matrix, int row, int col) {
    int i, j;

    double **derivative_matrix = (double**) malloc(row * sizeof (double));

    for (i = 0; i < row; i++)
        derivative_matrix[i] = (double*) malloc(col * sizeof (double));

    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            derivative_matrix[i][j] = matrix[i][j] * (1 - matrix[i][j]);
        }
    }

    return derivative_matrix;
}

double** matrix_transpose(double **matrix, int row, int col) {
    double **matrix_tranposed = (double**) malloc(col * sizeof (double));
    int i, j;

    for (i = 0; i < col; i++)
        matrix_tranposed[i] = (double*) malloc(row * sizeof (double));

    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            matrix_tranposed[j][i] = matrix[i][j];
        }
    }

    return matrix_tranposed;
}

double** hadamard_product(double ** matrix_1, double **matrix_2, int row, int col) {
    double **matrix_hadamard = (double**) malloc(col * sizeof (double));
    int i, j;

    for (i = 0; i < row; i++)
        matrix_hadamard[i] = (double*) malloc(row * sizeof (double));

    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            matrix_hadamard[i][j] = matrix_1[i][j] * matrix_2[i][j];
        }
    }

    return matrix_hadamard;
}

double** multiply_scalar(double **matrix, double scalar, int row, int col) {
    int i, j;

    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            matrix[i][j] = matrix[i][j] * scalar;
        }
    }

    return matrix;
}

double get_mean(double **matrix, int row, int col) {
    int i, j;
    double mean = 0;

    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            mean += matrix[i][j];
        }
    }
    
    mean /= (row *col);

    return mean;
}

double get_variance(double **matrix, int row, int col) {
    int i, j;
    double mean = get_mean(matrix, row, col);
    double var = 0;

    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            var += pow((matrix[i][j] - mean), 2.0);
        }
    }
    
    var /= (row *col);

    return var;
}

double get_standard_deviation(double **matrix, int row, int col) {
    double var = get_variance(matrix, row, col);
    
    return sqrt(var);
}