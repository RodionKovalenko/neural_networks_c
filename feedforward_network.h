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
        double ***targets_3d;
        double ****targets_4d;

        int **targets_int;
        int ***targets_int_3d;
        int ****targets_int_4d;
        int training_mode;
        // dataset for 1-dimensional array [number of record][record dimensions]
        // each row is separate record
        double **dataset;
        // for 2-dimensional array, first dimension is for number of records
        double ***dataset_3d;
        // for 3-dimensional array, e.g. for image data
        double ****dataset_4d;

        int **dataset_int;
        // for 2-dimensional array, first dimension is for number of records
        int ***dataset_3d_int;
        // for 3-dimensional array, e.g. for image data
        int ****dataset_4d_int;

        //input array dimensions, i.e. number of [records][colums][rows]
        int *input_dims;
        int num_records;
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
        int minibatch_size;
    } feedforward_network;

    feedforward_network init_ffn(
            double **inputs,
            int *input_dims,
            int input_num_records,
            int num_input_params,
            int n_h_layers,
            int n_h_neurons,
            int n_out_neurons,
            double **targets,
            double learning_rate
            );

    struct layer* init_layers(
            double **inputs,
            int *input_dims,
            int num_input_params,
            int n_h_layers,
            int n_h_neurons,
            int n_);

    feedforward_network fit(feedforward_network ffn, int num_iterations, int training_mode);
    void forward(feedforward_network ffn, int record_index);

#ifdef __cplusplus
}
#endif

#endif /* FEEDFORWARD_NETWORK_H */

