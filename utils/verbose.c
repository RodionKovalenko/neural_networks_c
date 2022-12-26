/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.c to edit this template
 */

#include <stdio.h>
#include <stdlib.h>
#include "../layer.h"
#include "../feedforward_network.h"

// generic implemenation of print_matrix
#define print_matrix(a, b, c) _Generic(a, double**:print_matrix_double, int**:print_matrix_int)(a, b, c)

void print_matrix_double(double **matrix, int rows, int columns) {
    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < columns; j++) {
            printf(" %f, ", *(*(matrix + i) + j));
        }
        printf("\n");
    }
}

void print_vector(double *matrix, int columns) {
    int j;

    for (j = 0; j < columns; j++) {
        printf(" %f, ", *(matrix + j));
    }
    printf("\n");
}

void print_matrix_int(int **matrix, int rows, int columns) {
    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < columns; j++) {
            printf(" %d, ", matrix[i][j]);
        }
        printf("\n");
    }
}

void print_layer(layer *_layer) {
    printf("layer index : %d \n", _layer->layer_index);
    printf("number of layer inputs and outputs: (%d => %d) \n", _layer->num_inputs, _layer->num_outputs);
    printf("number of layer rows and columns: (%d => %d) \n", _layer->num_input_rows, _layer->num_input_columns);


    if (_layer->weights != NULL) {
        printf("weight matrix size: number of outputs(%d, number of inputs %d) \n", _layer->num_outputs, _layer->num_inputs);

        print_matrix(_layer->weights, _layer->num_outputs, _layer->num_inputs);
    }

    if (_layer->previous_layer != NULL) {
        layer *tmp_layer = _layer;
        while (tmp_layer->previous_layer != NULL) {
            printf("previous layer is %s, index %d \n", tmp_layer->previous_layer->layer_name, tmp_layer->previous_layer->layer_index);

            tmp_layer = tmp_layer->previous_layer;
        }
    }
    if (_layer->next_layer != NULL) {
        layer *tmp_next_layer = _layer;
        while (tmp_next_layer->next_layer != NULL) {
            printf("next layer is %s, index %d \n", tmp_next_layer->next_layer->layer_name, tmp_next_layer->next_layer->layer_index);

            tmp_next_layer = tmp_next_layer->next_layer;
        }
    }


}

void print_network(feedforward_network network) {
    printf("verbose mode \n\n");
    int i, j;

    layer *layers = network.layers;
    int n_layers = network.n_h_layers;

    for (i = 0; i < n_layers + 2; i++) {
        layer *_layer = (layers + i);

        printf("\nlayer %d, name: %s\n", (i + 1), _layer->layer_name);
        if (_layer == NULL) {
            printf("layer is null");
            continue;
        }
        print_layer(_layer);
    }

}

void print_forward_updates(feedforward_network ffn, layer *_layer) {
    if (_layer->previous_layer != NULL && _layer->outputs != NULL) {
        printf("\nlayer index %d \n", _layer->layer_index);
        printf("number of inputs x outputs: %d x %d \n", _layer->num_inputs, _layer->num_outputs);
        printf("outputs : \n");
        print_matrix_double(_layer->outputs, _layer->num_outputs, ffn.input_dims[1]);
        printf("-------------------------------------------------------------\n");
    }
}
