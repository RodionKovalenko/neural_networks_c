/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.c to edit this template
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double ** get_random_normalized_weights(double **weight_matrix, int n_input, int n_output) {
    int i, j;

    for (i = 0; i < n_input; i++) {
        for (j = 0; j < n_output; j++) {
            weight_matrix[i][j] = 0;
        }
    }

    return weight_matrix;
}