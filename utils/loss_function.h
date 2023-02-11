/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.h to edit this template
 */

/* 
 * File:   loss_function.h
 * Author: rodion
 *
 * Created on January 21, 2023, 10:03 PM
 */

#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#ifdef __cplusplus
extern "C" {
#endif

    enum loss_function {
        MEAN_SQUARED_ERROR_LOSS,
        MAXIMUM_LIKELIHOOD_ESTIMATION_LOSS,
        MEAN_ABSOLUTE_ERROR_LOSS,
        HUBER_LOSS,
        BINARY_CROSS_ENTROPY_LOSS,
        CATEGORIAL_CROSS_ENTROPY_LOSS,
        HINGE_LOSS,
        COSINE_SIMILARITY_LOSS,
        LOG_COSH_LOSS,
        POISSON_LOSS
    };



#ifdef __cplusplus
}
#endif

#endif /* LOSS_FUNCTION_H */

