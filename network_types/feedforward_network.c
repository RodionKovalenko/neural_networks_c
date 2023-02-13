/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.c to edit this template
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "layer.h"
#include "network.h"
#include "feedforward_network.h"
#include "../utils/weight_initializer.h"
#include "../utils/verbose.h"
#include "../utils/array.h"
#include "../utils/math.h"
#include "../utils/activation.h"
#include "../utils/loss_function.h"
#include "../utils/optimizer.h"
#include "../utils/clear_memory.h"

static int verbose = 0;

void set_verbose(int value) {
    verbose = value;
}

network init_ffn(
        int *input_dims,
        int num_input_params,
        int input_num_records,
        int n_h_layers,
        int n_h_neurons,
        int n_out_neurons,
        double learning_rate,
        int activation,
        double bottleneck_value
        ) {


    // make n_h_layers odd if it is even
    if (n_h_layers % 2 == 0) {
        n_h_layers += 1;
    }

    layer *layers = init_layers(input_dims, num_input_params, n_h_layers, n_h_neurons, n_out_neurons, activation, bottleneck_value);

    network ffn = {
        .num_records = input_num_records,
        .learning_rate = learning_rate,
        .n_h_layers = n_h_layers,
        .n_h_neurons = n_h_neurons,
        .n_out_neurons = n_out_neurons,
        .minibatch_size = 15,
        .input_dims = input_dims,
        .layers = layers,
        .activation_type = activation,
        .errors = build_array(n_out_neurons, input_dims[1]),
        .is_gradient_checked = 0,
        .loss_function = MEAN_SQUARED_ERROR_LOSS,
        .optimizer = DEFAULT,
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
        int n_out_neurons,
        int activation,
        double bottleneck_value
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

    int m = n_h_layers + 2;
    int k;

    // hidden layers
    for (i = 1; i < n_h_layers + 1; i++) {
        layer *hidden_layer = (layer *) malloc(sizeof (struct layer));
        hidden_layer->num_inputs = layers[i - 1].num_outputs;
        hidden_layer->num_outputs = n_h_neurons;

        hidden_layer->num_input_rows = layers[i - 1].num_outputs;
        hidden_layer->num_input_columns = n_h_neurons;
        k = i + 1;

        // bottleneck of hidden layers 
        if (k > 2 && bottleneck_value != 0) {
            if ((m - k + 1) >= k) {
                hidden_layer->num_inputs = layers[i - 1].num_outputs;
                hidden_layer->num_outputs = (int) ((double) layers[i - 1].num_outputs / bottleneck_value) + 1;

                hidden_layer->num_input_rows = layers[i - 1].num_outputs;
                hidden_layer->num_input_columns = (int) ((double) layers[i - 1].num_outputs / bottleneck_value) + 1;
            } else {
                hidden_layer->num_inputs = layers[i - 1].num_outputs;
                hidden_layer->num_outputs = layers[m - k + 1].num_inputs;

                hidden_layer->num_input_rows = layers[i - 1].num_outputs;
                hidden_layer->num_input_columns = layers[m - k + 1].num_inputs;
            }
        }

        double **weight_matrix = build_array(hidden_layer->num_outputs, hidden_layer->num_inputs);

        init_random_weights(weight_matrix, hidden_layer->num_outputs, hidden_layer->num_inputs);
        char *layer_name = "hidden layer";

        hidden_layer->adam_A = build_array(hidden_layer->num_outputs, hidden_layer->num_inputs);
        hidden_layer->adam_B = build_array(hidden_layer->num_outputs, hidden_layer->num_inputs);
        hidden_layer->adam_A_bias = build_array(hidden_layer->num_outputs, input_dims[1]);
        hidden_layer->adam_B_bias = build_array(hidden_layer->num_outputs, input_dims[1]);

        hidden_layer->layer_name = layer_name;
        hidden_layer->layer_index = k;
        hidden_layer->weights = weight_matrix;

        hidden_layer->outputs = build_array(hidden_layer->num_outputs, input_dims[1]);
        hidden_layer->bias = build_array(hidden_layer->num_outputs, 1);
        hidden_layer->gradients = build_array(hidden_layer->num_outputs, input_dims[1]);
        hidden_layer->gradients_B = build_array(hidden_layer->num_outputs, 1);
        hidden_layer->gradients_W = build_array(hidden_layer->num_outputs, hidden_layer->num_inputs);
        hidden_layer->activation_type = activation;

        hidden_layer->previous_layer = &layers[i - 1];
        hidden_layer->next_layer = &layers[k];
        layers[i] = *hidden_layer;
        layers[i - 1].next_layer = hidden_layer;
    }

    if (n_h_layers == 0) {
        n_h_neurons = input_dims[2];
    }

    // output layer
    layer *output_layer = (layer *) malloc(sizeof (struct layer));
    double **weight_matrix = build_array(n_out_neurons, n_h_neurons);

    init_random_weights(weight_matrix, n_out_neurons, n_h_neurons);

    output_layer->weights = weight_matrix;

    output_layer->num_input_rows = layers[i - 1].num_outputs;
    output_layer->num_input_columns = n_out_neurons;
    output_layer->num_inputs = layers[i - 1].num_outputs;
    output_layer->num_outputs = n_out_neurons;
    output_layer->layer_name = "output layer";
    output_layer->layer_index = (i + 1);
    output_layer->previous_layer = &layers[i - 1];
    output_layer->outputs = build_array(output_layer->num_outputs, input_dims[1]);
    output_layer->bias = build_array(output_layer->num_outputs, 1);
    output_layer->gradients = build_array(output_layer->num_outputs, input_dims[1]);
    output_layer->gradients_B = build_array(output_layer->num_outputs, 1);
    output_layer->gradients_W = build_array(output_layer->num_outputs, output_layer->num_inputs);

    output_layer->adam_A = build_array(output_layer->num_outputs, output_layer->num_inputs);
    output_layer->adam_B = build_array(output_layer->num_outputs, output_layer->num_inputs);
    output_layer->adam_A_bias = build_array(output_layer->num_outputs, input_dims[1]);
    output_layer->adam_B_bias = build_array(output_layer->num_outputs, input_dims[1]);

    output_layer->activation_type = activation;

    layers[i] = *output_layer;
    layers[i - 1].next_layer = output_layer;

    return layers;
}

void forward(network ffn, double *data_X) {
    int l;
    double **layer_input, **outputs;
    layer *_layer, *_prev_layer;

    double **data_X_matrix = convert_vector_to_matrix(data_X, ffn.input_dims[2], 1);

    ffn.layers[0].outputs = data_X_matrix;

    for (l = 1; l < ffn.n_h_layers + 2; l++) {
        _layer = &ffn.layers[l];
        _prev_layer = _layer->previous_layer;

        if (_prev_layer != NULL) {
            layer_input = _prev_layer->outputs;
            outputs = _layer->outputs;

            if (_prev_layer->layer_index == 1) {
                outputs = apply_matrix_product(outputs, _layer->weights, data_X_matrix, _layer->num_outputs, ffn.input_dims[1], _layer->num_inputs);
            } else {
                outputs = apply_matrix_product(outputs, _layer->weights, layer_input, _layer->num_outputs, ffn.input_dims[1], _layer->num_inputs);
            }

            _layer->outputs = matrix_add_bias(outputs, _layer->bias, _layer->num_outputs, ffn.input_dims[1]);

            apply_activation(_layer, ffn);
        }
    }
}

// https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
void backward(network ffn, double *data_X, double *target_Y) {
    int l;
    layer *_layer;

    double **target_Y_matrix = convert_vector_to_matrix(target_Y, ffn.n_out_neurons, ffn.input_dims[1]);
    double **data_X_matrix = convert_vector_to_matrix(data_X, ffn.input_dims[2], ffn.input_dims[1]);

    // calculate gradients
    for (l = ffn.n_h_layers + 1; l > 0; l--) {
        _layer = &ffn.layers[l];

        if (l == (ffn.n_h_layers + 1)) {
            ffn.errors = get_errors(ffn, _layer->outputs, target_Y, _layer->num_outputs, ffn.input_dims[1]);
        }

        _layer->gradients = calculate_jacobi_matrix(&ffn, _layer, target_Y_matrix, data_X_matrix);
    }
}

double **get_errors(network ffn, double **output, double *target_Y, int n_row, int n_col) {
    int i, j;
    int normalizing_constant = ffn.minibatch_size;

    if (ffn.num_records < ffn.minibatch_size) {
        normalizing_constant = ffn.num_records;
    }

    for (i = 0; i < n_row; i++) {
        for (j = 0; j < n_col; j++) {
            switch (ffn.loss_function) {
                case MEAN_SQUARED_ERROR_LOSS:
                    ffn.errors[j][i] += pow((target_Y[j] - output[i][j]), 2.0) / (double) normalizing_constant;
            }
        }
    }

    return ffn.errors;
}

double **calculate_jacobi_matrix(network *ffn, layer *_layer, double **targetY, double **data_X) {
    int i, j, k, n;
    int n_row = _layer->num_outputs;
    int d = ffn->input_dims[1];
    layer *_next_layer = _layer->next_layer;
    layer *_prev_layer = _layer->previous_layer;
    double derivative;

    for (i = 0; i < n_row; i++) {
        for (k = 0; k < d; k++) {
            derivative = apply_deactivation_to_value(_layer, i, k, *ffn);

            if (_layer->layer_index == (ffn->n_h_layers + 2)) {
                _layer->gradients[i][k] += -2.0 * (targetY[i][k] - _layer->outputs[i][k]) * derivative;
            } else {
                for (n = 0; n < _next_layer->num_outputs; n++) {
                    _layer->gradients[i][k] += _next_layer->gradients[n][k] * _next_layer->weights[n][i] * derivative;
                }
            }

            _layer->gradients_B[i][0] += _layer->gradients[i][k];

            for (j = 0; j < _layer->num_inputs; j++) {
                if (_layer->layer_index == 2) {
                    _layer->gradients_W[i][j] += _layer->gradients[i][k] * data_X[j][k];
                } else {
                    _layer->gradients_W[i][j] += _layer->gradients[i][k] * _prev_layer->outputs[j][k];
                }
            }
        }
    }

    return _layer->gradients;
}

void update_weights(network ffn, int i_iteration) {
    int i, j, r, l, k;
    int normalizing_constant = ffn.minibatch_size;
    double p = 0.9999, pf = 0.9;
    double t = (double) i_iteration + 1;
    double prev_weight, prev_bias;
    double beta = 0.2;
    layer *_layer;

    if (ffn.num_records < ffn.minibatch_size) {
        normalizing_constant = ffn.num_records;
    }

    for (l = ffn.n_h_layers + 1; l >= 1; l--) {
        _layer = &ffn.layers[l];

        for (j = 0; j < _layer->num_outputs; j++) {
            for (i = 0; i < _layer->num_inputs; i++) {
                prev_weight = _layer->weights[j][i];
                switch (ffn.optimizer) {
                    case DEFAULT:
                        _layer->weights[j][i] -= ffn.learning_rate * _layer->gradients_W[j][i] / (double) normalizing_constant;
                        ffn.learning_rate *= 0.999;
                        break;
                    case ADAM:
                        update_weight_adam(&ffn, _layer, normalizing_constant, t, i, j, p, pf);
                        break;
                    default:
                        _layer->weights[j][i] -= ffn.learning_rate * _layer->gradients_W[j][i] / (double) normalizing_constant;
                        break;
                }

                _layer->weights[j][i] = beta * prev_weight + (1.0 - beta) * _layer->weights[j][i];
            }

            prev_bias = _layer->bias[j][0];

            switch (ffn.optimizer) {
                case DEFAULT:
                    _layer->bias[j][0] -= ffn.learning_rate * _layer->gradients_B[j][0] / (double) normalizing_constant;
                    break;
                case ADAM:
                    update_bias_adam(&ffn, _layer, normalizing_constant, t, i, j, p, pf);
                    break;
                default:
                    _layer->bias[j][0] -= ffn.learning_rate * _layer->gradients_B[j][0] / (double) normalizing_constant;
                    break;
            }

            _layer->bias[j][0] = beta * prev_bias + (1.0 - beta) * _layer->bias[j][0];
        }

        clear_array(_layer->gradients, _layer->num_outputs, ffn.input_dims[1]);
        clear_array(_layer->gradients_W, _layer->num_outputs, _layer->num_inputs);
        clear_array(ffn.errors, ffn.n_out_neurons, ffn.input_dims[1]);
        clear_array(_layer->gradients_B, _layer->num_outputs, 1);
    }

    if (verbose == 1) {

        printf("***********************************************************************\n");
        printf("weight are updated \n");
        printf("***********************************************************************\n\n");
    }
}

void update_weight_adam(network *ffn, layer *_layer, int normalizing_constant, double t, int i, int j, double p, double pf) {
    double learning_rate;

    ffn->learning_rate = ffn->learning_rate * (pow(1.0 - pow(p, t), 0.5) / (1.0 - pow(pf, t)));
    learning_rate = (ffn->learning_rate * _layer->adam_B[j][i]) / pow(_layer->adam_A[j][i] + 0.0001, 0.5);

    _layer->weights[j][i] -= learning_rate / (double) normalizing_constant;

    _layer->adam_A[j][i] = (p * _layer->adam_A[j][i] + (1.0 - p) * pow(_layer->gradients_W[j][i], 2.0));
    _layer->adam_B[j][i] = (pf * _layer->adam_B[j][i] + (1.0 - pf) * _layer->gradients_W[j][i]);
}

void update_bias_adam(network *ffn, layer *_layer, int normalizing_constant, double t, int i, int j, double p, double pf) {
    double learning_rate_bias;

    learning_rate_bias = (ffn->learning_rate * _layer->adam_B_bias[j][i]) / pow(_layer->adam_A_bias[j][i] + 0.0001, 0.5);

    _layer->bias[j][0] -= learning_rate_bias / (double) normalizing_constant;

    _layer->adam_A_bias[j][0] = (p * _layer->adam_A_bias[j][0] + (1.0 - p) * pow(_layer->gradients_B[j][0], 2.0));
    _layer->adam_B_bias[j][0] = (pf * _layer->adam_B_bias[j][i] + (1.0 - pf) * _layer->gradients_B[j][0]);
}

network fit(network ffn, double **data_X, double **target_Y, int num_iterations, int training_mode) {
    int i, j, r;
    int record_index;
    int minibach_index;
    double is_early_stop = 0;

    minibach_index = 0;
    for (i = 0; i < num_iterations && is_early_stop == 0; i++) {
        if (i % 50 == 0 || i == num_iterations - 1) {
            printf("processed: %f percent\n\n\n", ((double) (i + 1) / (double) num_iterations));
        }

        for (record_index = 0; record_index < ffn.num_records && is_early_stop == 0; record_index++) {
            forward(ffn, data_X[record_index]);
            backward(ffn, data_X[record_index], target_Y[record_index]);

            //print_network(ffn);
            if (i % 100 == 0 && record_index == ffn.num_records - 1) {
                // check the correctness of gradient on the first iteration
                check_gradient(&ffn, data_X[record_index], target_Y[record_index]);

                if (ffn.is_gradient_checked == 0) {
                    printf("Gradient is wrong. Break the training.\n");
                    is_early_stop = 1;
                    break;
                }
            }

            if (verbose == 1) {
                printf("======================================================= iteration index %d \n", i);
                printf("======================================================= record index %d \n", record_index);
                printf("target output \n");
                print_vector(target_Y[record_index], ffn.n_out_neurons);
                printf("network output \n");
                print_matrix_double(ffn.layers[ffn.n_h_layers + 1].outputs, ffn.n_out_neurons, ffn.input_dims[1]);
            }


            if (minibach_index != 0 && (minibach_index % ffn.minibatch_size == 0)) {
                printf("*********************************errors ***********************\n\n");
                print_matrix_double(ffn.errors, ffn.n_out_neurons, ffn.input_dims[1]);
                printf("*********************************errors ***********************\n\n");

                if (check_early_stopping(ffn) == 1) {
                    printf("======================================================= iteration index %d \n", i);
                    printf("EARLY STOPPING ACHIEVED \n");
                    is_early_stop = 1;
                    break;
                }
                update_weights(ffn, i);
                // reset minibatch index
                minibach_index = 0;
            }
            if (ffn.num_records > ffn.minibatch_size) {
                minibach_index++;
            }
        }

        // if number of records are smaller than batchsize than update weights after records iterations
        if (ffn.num_records < ffn.minibatch_size) {
            if (verbose == 0) {
                printf("*********************************errors ***********************\n");
                print_matrix_double(ffn.errors, ffn.n_out_neurons, ffn.input_dims[1]);
                printf("*********************************errors ***********************\n\n");
            }

            if (check_early_stopping(ffn) == 1 && (i != num_iterations - 1)) {
                printf("======================================================= iteration index %d \n", i);
                printf("EARLY STOPPING ACHIEVED \n");
                is_early_stop = 1;

                break;
            }
            update_weights(ffn, i);
        }
    }
}

int check_early_stopping(network ffn) {
    int i, j;

    int can_be_stopped = 0;

    for (i = 0; i < ffn.n_out_neurons; i++) {
        for (j = 0; j < ffn.input_dims[1]; j++) {
            if (ffn.errors[i][j] < 0.0001) {
                can_be_stopped = 1;
            } else {

                can_be_stopped = 0;
            }
        }
    }

    return can_be_stopped;
}

void check_gradient(network *ffn, double *data_X, double *target_Y) {
    if (ffn->is_gradient_checked == 1) {
        return;
    }
    // print_network(ffn);
    ffn->is_gradient_checked = 1;

    update_weights(*ffn, 2);

    forward(*ffn, data_X);
    backward(*ffn, data_X, target_Y);

    int i, j;
    layer _first_h_layer = ffn->layers[1];
    layer _output_layer = ffn->layers[ffn->n_h_layers + 1];
    double **original_outputs = build_array(ffn->n_out_neurons, ffn->input_dims[1]);

    double **outputs = _output_layer.outputs;
    double output_y1;
    double calculated_derivative_bias;
    double delta_x = 0.0000001;

    double output_y2;
    double calculated_derivative;
    double derivative;

    for (j = 0; j < ffn->n_out_neurons; j++) {
        for (i = 0; i < ffn->input_dims[1]; i++) {
            original_outputs[j][i] = outputs[j][i];
        }
    }

    printf("\n----------------------------------------- gradient check start--------------- \n\n\n");
    output_y1 = original_outputs[0][0];
    // test 1: output layer
    _output_layer.weights[0][0] = _output_layer.weights[0][0] + delta_x;
    forward(*ffn, data_X);

    outputs = _output_layer.outputs;

    output_y2 = outputs[0][0];
    calculated_derivative = _output_layer.gradients_W[0][0];
    derivative = ((pow((target_Y[0] - output_y2), 2.0) - pow((target_Y[0] - output_y1), 2.0)) / delta_x);

    printf("derivative of the first weight of output layer %f\n", derivative);
    printf("calculated derivative of first weight of output layer %f\n", calculated_derivative);

    if (fabs(fabs(derivative) - fabs(calculated_derivative)) < 0.0001) {
        printf("\n\n\nderivative of the output layer is correct \n\n\n");
    } else {
        printf("GRADIENT IS NOT CORRECT\n");
        ffn->is_gradient_checked = 0;
    }
    _output_layer.weights[0][0] = _output_layer.weights[0][0] - delta_x;

    // test 2: first hidden layer
    _first_h_layer.weights[0][0] = _first_h_layer.weights[0][0] + delta_x;
    forward(*ffn, data_X);

    outputs = _output_layer.outputs;
    derivative = 0.0;

    for (j = 0; j < _output_layer.num_outputs; j++) {
        for (i = 0; i < ffn->input_dims[1]; i++) {
            output_y2 = outputs[j][i];
            derivative += ((pow((target_Y[j] - output_y2), 2.0) - pow((target_Y[j] - original_outputs[j][i]), 2.0)) / delta_x);
        }
    }

    calculated_derivative = _first_h_layer.gradients_W[0][0];
    printf("derivative of the first weight of first hidden layer %f\n", derivative);
    printf("calculated derivative of first weight of hidden layer %f\n", calculated_derivative);

    if (fabs(fabs(derivative) - fabs(calculated_derivative)) < 0.0001) {
        printf("\n\n\nderivative of the hidden layer is correct \n\n\n");
    } else {
        printf("GRADIENT IS NOT CORRECT\n");
        ffn->is_gradient_checked = 0;
    }

    _first_h_layer.weights[0][0] = _first_h_layer.weights[0][0] - delta_x;

    // test 3: gradient of bias in output layer
    _output_layer.bias[0][0] = _output_layer.bias[0][0] + delta_x;
    forward(*ffn, data_X);

    outputs = _output_layer.outputs;
    output_y2 = outputs[0][0];
    derivative = ((pow((target_Y[0] - output_y2), 2.0) - pow((target_Y[0] - output_y1), 2.0)) / delta_x);
    calculated_derivative_bias = _output_layer.gradients_B[0][0];

    printf("bias derivative of the first weight of output layer %f\n", derivative);
    printf("bias calculated derivative of first weight of output layer %f\n", calculated_derivative_bias);

    _output_layer.bias[0][0] = _output_layer.bias[0][0] - delta_x;

    if (fabs(fabs(derivative) - fabs(calculated_derivative_bias)) < 0.0001) {
        printf("\n\n\nbias derivative of the output layer is correct \n\n\n");
    } else {
        printf("BIAS GRADIENT IS NOT CORRECT\n");
        ffn->is_gradient_checked = 0;
    }

    // test 4: gradient of bias in first hidden layer
    _first_h_layer.bias[0][0] = _first_h_layer.bias[0][0] + delta_x;
    forward(*ffn, data_X);

    outputs = _output_layer.outputs;
    calculated_derivative_bias = _first_h_layer.gradients_B[0][0];
    derivative = 0.0;

    for (j = 0; j < _output_layer.num_outputs; j++) {
        for (i = 0; i < ffn->input_dims[1]; i++) {
            output_y2 = outputs[j][i];
            derivative += ((pow((target_Y[j] - output_y2), 2.0) - pow((target_Y[j] - original_outputs[j][i]), 2.0)) / delta_x);
        }
    }

    printf("bias derivative of the first weight of first hidden layer %f\n", derivative);
    printf("bias calculated derivative of first weight of hidden layer %f\n", calculated_derivative_bias);
    if (fabs(fabs(derivative) - fabs(calculated_derivative_bias)) < 0.0001) {
        printf("\n\n\n bias derivative of the first hidden layer is correct \n\n\n");
    } else {

        printf("BIAS GRADIENT IS NOT CORRECT\n");
        ffn->is_gradient_checked = 0;
    }

    _first_h_layer.bias[0][0] = _first_h_layer.bias[0][0] - delta_x;
    forward(*ffn, data_X);

    printf("\n----------------------------------------- gradient check end -------------- \n\n\n");
}

