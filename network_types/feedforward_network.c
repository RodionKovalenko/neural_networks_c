/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.c to edit this template
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../layer.h"
#include "../feedforward_network.h"
#include "../weight_initializer.h"
#include "../verbose.h"
#include "../array.h"
#include "../math.h"
#include "../activation.h"

static int verbose = 0;

void set_verbose(int value) {
    verbose = value;
}

feedforward_network init_ffn(
        int *input_dims,
        int num_input_params,
        int input_num_records,
        int n_h_layers,
        int n_h_neurons,
        int n_out_neurons,
        double learning_rate,
        int activation
        ) {

    layer *layers = init_layers(input_dims, num_input_params, n_h_layers, n_h_neurons, n_out_neurons);

    feedforward_network ffn = {
        .num_records = input_num_records,
        .learning_rate = learning_rate,
        .n_h_layers = n_h_layers,
        .n_h_neurons = n_h_neurons,
        .n_out_neurons = n_out_neurons,
        .minibatch_size = 15,
        .input_dims = input_dims,
        .layers = layers,
        .activation_type = activation,
    };

    if (verbose == 1) {
        print_network(ffn);
    }

    return ffn;
}

struct layer* init_layers(
        int *input_dims,
        int num_input_params,
        int n_h_layers,
        int n_h_neurons,
        int n_out_neurons
        ) {
    int i;

    layer *layers = (layer *) malloc(sizeof (struct layer) * (n_h_layers + 2));
    layer *input_layer = (layer *) malloc(sizeof (struct layer));

    input_layer->layer_name = "input layer";
    layers[0] = *input_layer;

    // input layer 0
    for (i = 0; i < num_input_params; i++) {
        if (i == 2) {
            layers[0].num_inputs = input_dims[i];
            layers[0].num_input_rows = input_dims[i];
            layers[0].num_outputs = input_dims[i];
            layers[0].num_input_columns = input_dims[i];
        }
    }
    layers[0].layer_index = 1;

    // hidden layers
    for (i = 1; i < n_h_layers + 1; i++) {
        layer *hidden_layer = (layer *) malloc(sizeof (struct layer));
        hidden_layer->num_inputs = layers[i - 1].num_outputs;
        hidden_layer->num_outputs = n_h_neurons;

        hidden_layer->num_input_rows = layers[i - 1].num_outputs;
        hidden_layer->num_input_columns = n_h_neurons;

        double **weight_matrix = build_array(hidden_layer->num_outputs, hidden_layer->num_inputs);

        init_random_weights(weight_matrix, hidden_layer->num_outputs, hidden_layer->num_inputs);
        char *layer_name = "hidden layer";

        hidden_layer->layer_name = layer_name;
        hidden_layer->layer_index = (i + 1);
        hidden_layer->weights = weight_matrix;

        hidden_layer->outputs = build_array(hidden_layer->num_outputs, input_dims[1]);
        hidden_layer->gradients = build_array(hidden_layer->num_outputs, input_dims[1]);
        hidden_layer->gradients_W = build_array(hidden_layer->num_outputs, hidden_layer->num_inputs);

        hidden_layer->previous_layer = &layers[i - 1];
        hidden_layer->next_layer = &layers[i + 1];
        layers[i] = *hidden_layer;
        layers[i - 1].next_layer = hidden_layer;
    }

    // output layer
    layer *output_layer = (layer *) malloc(sizeof (struct layer));
    double **weight_matrix = build_array(n_out_neurons, n_h_neurons);

    init_random_weights(weight_matrix, n_out_neurons, n_h_neurons);

    output_layer->weights = weight_matrix;

    output_layer->num_input_rows = layers[i - 1].num_outputs;
    output_layer->num_input_columns = n_out_neurons;
    output_layer->num_inputs = layers[i - 1].num_inputs;
    output_layer->num_outputs = n_out_neurons;
    output_layer->layer_name = "output layer";
    output_layer->layer_index = (i + 1);
    output_layer->previous_layer = &layers[i - 1];
    output_layer->outputs = build_array(output_layer->num_outputs, input_dims[1]);
    output_layer->gradients = build_array(output_layer->num_outputs, input_dims[1]);
    output_layer->errors = build_array(output_layer->num_outputs, input_dims[1]);
    output_layer->gradients_W = build_array(output_layer->num_outputs, input_dims[1]);

    layers[i] = *output_layer;
    layers[i - 1].next_layer = output_layer;

    return layers;
}

void forward(feedforward_network ffn, double *data_X) {
    int i, j, r, l, k;
    double **layer_input, **outputs;
    layer *_layer, *_prev_layer;

    for (l = 1; l < ffn.n_h_layers + 2; l++) {
        _layer = &ffn.layers[l];
        _prev_layer = _layer->previous_layer;

        if (_prev_layer != NULL) {
            layer_input = _prev_layer->outputs;
            outputs = _layer->outputs;

            clear_array(outputs, _layer->num_outputs, ffn.input_dims[1]);

            // forward output to the next layer
            for (j = 0; j < _layer->num_outputs; j++) {
                for (i = 0; i < _layer->num_inputs; i++) {
                    for (int k = 0; k < ffn.input_dims[1]; k++) {
                        if (_prev_layer->layer_index == 1) {
                            outputs[j][k] += _layer->weights[j][i] * data_X[i];
                        } else {
                            outputs[j][k] += _layer->weights[j][i] * layer_input[i][k];
                        }
                    }
                }
            }

            _layer->outputs = outputs;
            apply_activation(_layer, ffn);
        }

        if (_prev_layer != NULL && outputs != NULL && verbose == 1) {
            //   print_layer(_layer);
        }
    }
}

double **apply_activation(layer *_layer, feedforward_network ffn) {
    switch (ffn.activation_type) {
        case SIGMOID:
            return sigmoid_to_matrix(_layer->outputs, _layer->num_outputs, ffn.input_dims[1]);
        case IDENTITY:
            return _layer->outputs;
        case SOFTMAX:
            break;
        case BINARY:
            break;
        case TANH:
            break;
        case RELU:
            break;
        case SWISH:
            break;
        case GELU:
            break;
        case SELU:
            break;
    }
}

double **apply_deactivation(layer *_layer, feedforward_network ffn) {
    switch (ffn.activation_type) {
        case SIGMOID:
            return sigmoid_to_matrix(_layer->outputs, _layer->num_outputs, ffn.input_dims[1]);
        case IDENTITY:
            return _layer->outputs;
        case SOFTMAX:
            break;
        case BINARY:
            break;
        case TANH:
            break;
        case RELU:
            break;
        case SWISH:
            break;
        case GELU:
            break;
        case SELU:
            break;
    }
}

double apply_deactivation_to_value(double value, feedforward_network ffn) {
    switch (ffn.activation_type) {
        case SIGMOID:
            return sigmoid_derivate_to_value(value);
        case IDENTITY:
            return value;
        case SOFTMAX:
            break;
        case BINARY:
            break;
        case TANH:
            break;
        case RELU:
            break;
        case SWISH:
            break;
        case GELU:
            break;
        case SELU:
            break;
    }
}

void backward(feedforward_network ffn, double *data_X, double *target_Y) {
    int i, j, r, l, k;
    double value_der;
    layer *_layer, *_prev_layer, *_next_layer;

    // calculate gradients
    for (l = ffn.n_h_layers + 1; l > 0; l--) {
        _layer = &ffn.layers[l];
        _next_layer = _layer->next_layer;
        _prev_layer = _layer->previous_layer;

        for (j = 0; j < _layer->num_outputs; j++) {
            for (i = 0; i < ffn.input_dims[1]; i++) {
                value_der = apply_deactivation_to_value(_layer->outputs[j][i], ffn);
                if (l == (ffn.n_h_layers + 1)) {
                    _layer->errors[j][i] += fabs(target_Y[j] - _layer->outputs[j][i]);
                    _layer->gradients[j][i] += (target_Y[j] - _layer->outputs[j][i]);
                } else {
                    for (k = 0; k < _next_layer->num_outputs; k++) {
                        _layer->gradients[j][i] += _next_layer->gradients[k][i] * _next_layer->weights[k][j];
                        //_layer->gradients[j][i] += _next_layer->gradients[k][i];
                    }
                    _layer->gradients[j][i] *= value_der;
                }

                for (k = 0; k < _layer->num_inputs; k++) {
                    if (l == 1) {
                        _layer->gradients_W[j][k] += _layer->gradients[j][i] * data_X[k];
                    } else {
                        _layer->gradients_W[j][k] += _layer->gradients[j][i] * _prev_layer->outputs[k][i];
                    }
                }
            }
        }

        //        if (verbose == 1) {
        //            printf("-------------------------backwards------------------------------- \n");
        //            print_layer(_layer);
        //            printf("-------------------------backwards------------------------------- \n\n");
        //        }
    }
}

void update_weights(feedforward_network ffn) {
    int i, j, r, l, k;
    layer *_layer;

    for (l = ffn.n_h_layers + 1; l >= 1; l--) {
        _layer = &ffn.layers[l];

        for (j = 0; j < _layer->num_outputs; j++) {
            for (i = 0; i < _layer->num_inputs; i++) {
                _layer->weights[j][i] += (ffn.learning_rate * _layer->gradients_W[j][i]) / (double) ffn.num_records;
            }
        }

        //print_layer(_layer);

        clear_array(_layer->gradients, _layer->num_outputs, ffn.input_dims[1]);
        clear_array(_layer->gradients_W, _layer->num_outputs, _layer->num_inputs);

        if (_layer->errors != NULL) {
            clear_array(_layer->errors, ffn.n_out_neurons, ffn.input_dims[1]);
        }
    }

    if (verbose == 1) {
        printf("***********************************************************************\n");
        printf("weight are updated \n");
        printf("***********************************************************************\n\n");
    }
}

feedforward_network fit(feedforward_network ffn, double **data_X, double **target_Y, int num_iterations, int training_mode) {
    int i, j, r;
    int record_index;
    int minibach_index;

    minibach_index = 0;
    for (i = 0; i < num_iterations; i++) {
        for (record_index = 0; record_index < ffn.num_records; record_index++) {
            forward(ffn, data_X[record_index]);
            backward(ffn, data_X[record_index], target_Y[record_index]);

            if (verbose == 1) {
                printf("======================================================= record index %d \n", record_index);
                printf("======================================================= iteration index %d \n", i);
                printf("target output \n");
                print_vector(target_Y[record_index], ffn.n_out_neurons);
                printf("network output \n");
                print_matrix_double(ffn.layers[ffn.n_h_layers + 1].outputs, ffn.n_out_neurons, ffn.input_dims[1]);
            }

            if (ffn.num_records > ffn.minibatch_size) {
                minibach_index++;
            }
            if (minibach_index != 0 && (minibach_index % ffn.minibatch_size == 0)) {
                printf("*********************************errors ***********************\n\n");
                print_matrix_double(ffn.layers[ffn.n_h_layers + 1].errors, ffn.n_out_neurons, ffn.input_dims[1]);
                printf("*********************************errors ***********************\n\n");

                update_weights(ffn);
                // reset minibatch index
                minibach_index = 0;
            }
        }

        // if number of records are smaller than batchsize than update weights after records iterations
        if (ffn.num_records < ffn.minibatch_size) {
            printf("*********************************errors ***********************\n");
            print_matrix_double(ffn.layers[ffn.n_h_layers + 1].errors, ffn.n_out_neurons, ffn.input_dims[1]);
            printf("*********************************errors ***********************\n\n");

            update_weights(ffn);
        }
    }
}

void clear_network(feedforward_network ffn) {
    int l = 0;
    layer *_layer;

    for (l = ffn.n_h_layers + 1; l >= 1; l--) {
        _layer = &ffn.layers[l];
        if (ffn.layers[l].errors != NULL) {
            clear_matrix_memory(ffn.layers[l].errors, ffn.n_out_neurons);
        }

        free(_layer->activation_type);
        free(_layer->next_layer);

        clear_matrix_memory(_layer->gradients, _layer->num_outputs);
        clear_matrix_memory(_layer->weights, _layer->num_outputs);
        clear_matrix_memory(_layer->outputs, _layer->num_outputs);
    }

    free(ffn.layers);
}

void clear_matrix_memory(double **matrix, int rows) {
    int i;

    for (i = 0; i < rows; i++) {
        free(matrix[i]);
    }

    free(matrix);
}