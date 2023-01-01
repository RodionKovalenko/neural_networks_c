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


#ifdef __cplusplus
}
#endif

#endif /* ACTIVATION_H */

