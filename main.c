/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/main.c to edit this template
 */

/* 
 * File:   main.c
 * Author: rodion
 *
 * Created on October 22, 2022, 7:36 PM
 */

#include <stdio.h>
#include <stdlib.h>
#include<time.h>
#include<string.h>
#include <math.h>
#include "activation.h"
#include <float.h>
#include <limits.h>
#include "verbose.h"
#include "layer.h"
#include "feedforward_network.h"
#include "array.h"

double** get_input_matrix(int input_r, int input_c) {
    double **input_matrix = (double**) malloc(input_r * (sizeof (double)));
    int i, j;

    for (i = 0; i < input_r; i++) {
        input_matrix[i] = (double*) malloc(input_c * sizeof (double));
    }

    input_matrix[0][0] = 0.5;
//    input_matrix[0][1] = 0.0;
//
//    input_matrix[1][0] = 1.0;
//    input_matrix[1][1] = 0.0;

//    input_matrix[2][0] = 1.0;
//    input_matrix[2][1] = 1.0;
//
//    input_matrix[3][0] = 1.0;
//    input_matrix[3][1] = 0.0;

    return input_matrix;
}

double** get_target_matrix(int input_r, int input_c) {
    double **input_matrix = (double**) malloc(input_r * sizeof (double));
    int i, j;

    for (i = 0; i < input_r; i++) {
        input_matrix[i] = (double*) malloc(input_c * sizeof (double));
    }

    input_matrix[0][0] = 0.0;
    input_matrix[0][1] = 1.0;
//    input_matrix[2][0] = 0.0;
//    input_matrix[3][0] = 1.0;

    return input_matrix;
}

void testnetwork() {
    // number of data records
    int input_num_records = 1;
    // number of dimensions in record
    int input_r = 1;
    int input_c = 1;

    int n_h_layers = 1;
    int n_h_neurons = 1;
    int n_out_neurons = 2;

    int num_dim[] = {input_num_records, input_r, input_c};
    int num_dim_params = sizeof (num_dim) / sizeof (int);

    double learning_rate = 0.003;
    int num_iterations = 1;
    int training_mode = 0;

    // one-dimensional training and target dataset 
    double **data_X = get_input_matrix(input_num_records, input_c);
    double **target_Y = get_target_matrix(input_num_records, n_out_neurons);

    set_verbose(1);
    printf("start");

    feedforward_network feedforward_net = init_ffn(
            num_dim,
            num_dim_params,
            input_num_records,
            n_h_layers,
            n_h_neurons,
            n_out_neurons,
            learning_rate,
            TANH
            );
    
    feedforward_net.layers[1].weights[0][0] = 0.5;
    feedforward_net.layers[2].weights[0][0] = 0.2;
    feedforward_net.layers[2].weights[1][0] = 0.1;
    
    fit(feedforward_net, data_X, target_Y, num_iterations, training_mode);
    clear_network(feedforward_net);
    clear_matrix_memory(data_X, input_num_records);
    clear_matrix_memory(target_Y, input_num_records);
}

int main(int argc, char** argv) {
    testnetwork();

    return (EXIT_SUCCESS);
}

