/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cFiles/file.h to edit this template
 */

/* 
 * File:   array.h
 * Author: rodion
 *
 * Created on December 23, 2022, 12:48 PM
 */

#ifndef ARRAY_H
#define ARRAY_H

#ifdef __cplusplus
extern "C" {
#endif

    double** build_array(int n_row, int n_col);
    double*** build_array_3d(int n_records, int n_row, int n_col);
    double** clear_array(double **array, int n_row, int n_col);

#ifdef __cplusplus
}
#endif

#endif /* ARRAY_H */

