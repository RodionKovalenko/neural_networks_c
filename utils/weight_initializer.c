/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.c to edit this template
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double ** init_random_weights(double **weight_matrix, int n_output, int n_input) {
    int i, j;

    double min = -pow((6.0 / (double) (n_input + n_output)), 0.5);
    double max = pow((6.0 / (double) (n_input + n_output)), 0.5);

    for (i = 0; i < n_output; i++) {
        for (j = 0; j < n_input; j++) {
            weight_matrix[i][j] = (double) rand() * (max - min) / (double) RAND_MAX + min;
            //weight_matrix[i][j] = 0.3;
        }
    }
    
    return weight_matrix;
}