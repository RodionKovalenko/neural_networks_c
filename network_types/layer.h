/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.h to edit this template
 */

/* 
 * File:   layer.h
 * Author: rodion
 *
 * Created on December 18, 2022, 5:37 PM
 */

#ifndef LAYER_H
#define LAYER_H

#ifdef __cplusplus
extern "C" {
#endif
    void set_verbose(int verbose);

    typedef struct layer {
        double **weights;
        double **prev_layer_weights;
        // layer output in 2_d: [record_index][input_dimension]
        double **outputs;
        double ***layer_sequence_outputs;
        double **errors;
        double **bias;
        double **gradients;
        double **gradients_prev;
        double **gradients_B;
        double **adam_A_bias;
        double **adam_B_bias;
        double **gradients_W;
        double **gradients_W_prev;
        double **adam_A;
        double **adam_B;
        struct layer *previous_layer;
        struct layer *next_layer;
        int activation_type;
        int num_inputs;
        int num_outputs;
        int num_input_rows;
        int num_input_columns;
        char *layer_name;
        // beginning with 1
        int layer_index;
    } layer;


#ifdef __cplusplus
}
#endif

#endif /* LAYER_H */

