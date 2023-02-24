/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.h to edit this template
 */

/* 
 * File:   activation.h
 * Author: rodion
 *
 * Created on October 23, 2022, 4:58 PM
 */

#ifndef ACTIVATION_H
#define ACTIVATION_H

#ifdef __cplusplus
extern "C" {
#endif
    
#include "../network_types/layer.h"
#include "../network_types/network.h"

    enum activation {
        SIGMOID,
        IDENTITY,
        TANH,
        RELU,
        LEAKY_RELU,
        SOFTMAX,
        MULTIPLY_TWO,
        SWISH,
        GELU,
        SELU
    };

    double **apply_activation(layer *_layer, network ffn);
    double **apply_deactivation(layer *_layer, network ffn);
    double apply_deactivation_to_value(layer *_layer, int row, int column, network ffn);
    double apply_deactivation_to_value_rnn(layer *_layer, int row, int column, int s, network ffn);

#ifdef __cplusplus
}
#endif

#endif /* ACTIVATION_H */

