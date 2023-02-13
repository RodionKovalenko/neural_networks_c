/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.c to edit this template
 */
#include <stdlib.h>
#include <stdio.h>
#include "../network_types/network.h"
#include "../network_types/layer.h"

void clear_matrix_memory_3d(double ***matrix, int d1, int d2) {
     int i, j;

    for (i = 0; i < d1; i++) {
        for (j = 0; j < d2; j++) {
              free(matrix[i][j]);
        }
        free(matrix[i]);
    }

    free(matrix);
}

void clear_matrix_memory(double **matrix, int rows) {
    int i;

    for (i = 0; i < rows; i++) {
        free(matrix[i]);
    }

    free(matrix);
}

void clear_network(network ffn) {
    int l = 0;
    layer *_layer;

    for (l = ffn.n_h_layers + 1; l >= 1; l--) {
        _layer = &ffn.layers[l];
        if (ffn.layers[l].errors != NULL) {

            clear_matrix_memory(ffn.layers[l].errors, ffn.n_out_neurons);
        }

        free(_layer->next_layer);

        clear_matrix_memory(_layer->gradients, _layer->num_outputs);
        clear_matrix_memory(_layer->gradients_W, _layer->num_outputs);
        clear_matrix_memory(_layer->bias, _layer->num_outputs);

        clear_matrix_memory(_layer->weights, _layer->num_outputs);
        clear_matrix_memory(_layer->outputs, _layer->num_outputs);
    }

    free(ffn.layers);
}

