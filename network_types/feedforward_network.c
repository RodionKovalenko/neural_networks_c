/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.c to edit this template
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include "../layer.h"
#include "../feedforward_network.h"
#include "../weight_initializer.h"


static int verbose = 0;

void set_verbose(int value) {
    verbose = value;
}

feedforward_network init_ffn(
        double **inputs,
        int *input_dims,
        int num_input_params,
        int n_h_layers,
        int n_h_neurons,
        int n_out_neurons,
        double learning_rate
        ) {

    struct layer *layers = init_layers(inputs, input_dims, num_input_params, n_h_layers, n_h_neurons, n_out_neurons);

    struct feedforward_network ffn = {
        .learning_rate = learning_rate,
        .n_h_layers = n_h_layers,
        .n_h_neurons = n_h_neurons,
        .n_out_neurons = n_out_neurons,
        .targets = NULL,
        .layers = layers
    };

    if (verbose == 1) {
        print_network(ffn);
    }

    return ffn;
}

double** build_array(int n_row, int n_col) {
    int i, j;
    double **result_matrix = (double**) malloc(n_row * sizeof (double));

    for (i = 0; i < n_row; i++) {
        result_matrix[i] = (double*) malloc(n_col * sizeof (double));
    }

    return result_matrix;
}

struct layer* init_layers(
        double **inputs,
        int *input_dims,
        int num_input_params,
        int n_h_layers,
        int n_h_neurons,
        int n_out_neurons
        ) {
    int i;

   layer *layers = (layer *) malloc(sizeof (struct layer) * (n_h_layers + 2));

    layers->inputs = inputs;

    layer *input_layer = (layer *) malloc(sizeof (struct layer));
    
    layers[0] = *input_layer;
    // input layer 0
    for (i = 0; i < num_input_params; i++) {
        if (i == 0) {
            layers[0].num_inputs = input_dims[i];
            layers[0].num_input_rows = input_dims[i];
        }
        if (i == 1) {
            layers[0].num_outputs = input_dims[i];
            layers[0].num_input_columns = input_dims[i];
        }
    }

    // hidden layers
    for (i = 1; i < n_h_layers; i++) {
        layer *hidden_layer = (layer *) malloc(sizeof (struct layer));
        hidden_layer->num_inputs = layers[i - 1].num_outputs;
        hidden_layer->num_outputs = n_h_neurons;
        hidden_layer->num_input_rows = layers[i - 1].num_outputs;
        hidden_layer->num_input_columns = n_h_neurons;

        hidden_layer->num_outputs = n_h_neurons;

        double **weight_matrix = build_array(n_h_neurons, hidden_layer->num_input_rows);

        hidden_layer->weights = weight_matrix;
        layers[i] = *hidden_layer;
        layers[i - 1].next_layer = hidden_layer;
        hidden_layer->previous_layer = &layers[i - 1];
    }

    // output layer
    layer *output_layer = (layer *) malloc(sizeof (struct layer));
    double **weight_matrix = build_array(n_out_neurons, n_h_neurons);

    output_layer->weights = weight_matrix;
    layers[i].next_layer = output_layer;
    output_layer->previous_layer = &layers[i];

    return layers;
}

