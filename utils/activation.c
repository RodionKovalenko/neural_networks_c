/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.c to edit this template
 */

#include "activation.h"
#include "math.h"

double **apply_activation(layer *_layer, network ffn) {
    switch (ffn.activation_type) {
        case SIGMOID:
            return sigmoid_to_matrix(_layer->outputs, _layer->num_outputs, ffn.input_dims[1]);
        case IDENTITY:
            return _layer->outputs;
        case SOFTMAX:
            // softmax can be applied only to output layer
            if (_layer->layer_index == (ffn.n_h_layers + 2)) {
                return softmax_to_matrix(_layer->outputs, _layer->num_outputs, ffn.input_dims[1]);
            } else {
                return sigmoid_to_matrix(_layer->outputs, _layer->num_outputs, ffn.input_dims[1]);
            }
            break;
        case TANH:
            return tanh_to_matrix(_layer->outputs, _layer->num_outputs, ffn.input_dims[1]);
        case RELU:
            return relu_to_matrix(_layer->outputs, _layer->num_outputs, ffn.input_dims[1]);
        case LEAKY_RELU:
            return leaky_relu_to_matrix(_layer->outputs, _layer->num_outputs, ffn.input_dims[1]);
        case MULTIPLY_TWO:
            return multiply_two_to_matrix(_layer->outputs, _layer->num_outputs, ffn.input_dims[1]);
        case SWISH:
            break;
        case GELU:
            break;
        case SELU:
            break;
    }
}

double apply_deactivation_to_value(layer *_layer, int row, int column, network ffn) {
    double value = _layer->outputs[row][column];

    switch (ffn.activation_type) {
        case SIGMOID:
            return sigmoid_derivate_to_value(value);
        case IDENTITY:
            return 1.0;
        case SOFTMAX:
            // softmax can be applied only to output layer
            if (_layer->layer_index == (ffn.n_h_layers + 2)) {
                return softmax_derivate_to_value(_layer->outputs, row, column);
            } else {
                return sigmoid_derivate_to_value(value);
            }
        case TANH:
            return tanh_derivative_to_value(value);
            break;
        case RELU:
            return relu_derivative_to_value(value);
        case LEAKY_RELU:
            return leaky_relu_derivative_to_value(value);
        case MULTIPLY_TWO:
            return multiply_two_derivative_to_value(value);
        case SWISH:
            break;
        case GELU:
            break;
        case SELU:
            break;
    }
}

double apply_deactivation_to_value_rnn(layer *_layer, int row, int column, int s, network ffn) {
    double value = _layer->layer_sequence_outputs[s][row][column];
    
    switch (ffn.activation_type) {
        case SIGMOID:
            return sigmoid_derivate_to_value(value);
        case IDENTITY:
            return 1.0;
        case SOFTMAX:
            // softmax can be applied only to output layer
            if (_layer->layer_index == (ffn.n_h_layers + 2)) {
                return softmax_derivate_to_value(_layer->outputs, row, column);
            } else {
                return sigmoid_derivate_to_value(value);
            }
        case TANH:
            return tanh_derivative_to_value(value);
            break;
        case RELU:
            return relu_derivative_to_value(value);
        case LEAKY_RELU:
            return leaky_relu_derivative_to_value(value);
        case MULTIPLY_TWO:
            return multiply_two_derivative_to_value(value);
        case SWISH:
            break;
        case GELU:
            break;
        case SELU:
            break;
    }
}