/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.h to edit this template
 */

/* 
 * File:   optimizer.h
 * Author: rodion
 *
 * Created on January 22, 2023, 7:25 PM
 */

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#ifdef __cplusplus
extern "C" {
#endif

     enum optimizer {
        ADAM,
        ADAM_GRAD,
        NESTOROV,
        RMS_PROP,
        RMS_PROP_NESTEROV,
        ADA_DELTA,
        GRADIENT_CLIPPING,
        HESSIAN_FREE_OPT,
        QUASI_NEWTON,
        POLYAK_AVERAGING
    };
    


#ifdef __cplusplus
}
#endif

#endif /* OPTIMIZER_H */

