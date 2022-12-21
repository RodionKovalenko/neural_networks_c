/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.h to edit this template
 */

/* 
 * File:   feedforward_network.h
 * Author: rodion
 *
 * Created on December 18, 2022, 6:40 PM
 */

#ifndef FEEDFORWARD_NETWORK_H
#define FEEDFORWARD_NETWORK_H

#ifdef __cplusplus
extern "C" {
#endif
    void set_verbose(int verbose);

    typedef struct feedforward_network {
        double **targets;
        int training_mode;
        double **inputs;
        //input array dimensions, i.e. number of row/colums/etc 
        //e.g., {2, 1}: number of rows 2, number of colums: 1
        int *input_dims;
        // number of input params, e.g 2 (rows and columns)
        int num_input_params;
        int n_input_rows;
        int n_input_columns;
        // number of hidden layers
        int n_h_layers;
        // number of hidden neurons in a hidden layer
        int n_h_neurons;
        // number of output neurons in the last output layer
        int n_out_neurons;
        // learning rate/
        double learning_rate;
        struct layer *layers;
    } feedforward_network;


    feedforward_network init_ffn(
            double **inputs,
            int *input_dims,
            int num_input_params,
            int n_h_layers,
            int n_h_neurons,
            int n_out_neurons,
            double learning_rate
            );

    struct layer* init_layers(
            double **inputs,
            int *input_dims,
            int num_input_params,
            int n_h_layers,
            int n_h_neurons,
            int n_);


#ifdef __cplusplus
}
#endif

#endif /* FEEDFORWARD_NETWORK_H */

