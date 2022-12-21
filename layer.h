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

 typedef struct layer {
        double **weights;
        double **inputs;
        double **gradients;
        double **prev_gradients;
        struct layer *previous_layer;
        struct layer *next_layer;
        int num_inputs;
        int num_outputs;
        int num_input_rows;
        int num_input_columns;
        char *activation_type;
    } layer;


#ifdef __cplusplus
}
#endif

#endif /* LAYER_H */

