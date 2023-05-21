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
#include<string.h>

#define BUFFER_SIZE 200

typedef struct {
    char *buffer;
} file_entry;

int main(int argc, char** argv) {
    int i, c, totalRead = 0;

    char *arg_value;
    file_entry real;

    for (i = 0; i < argc; i++) {
        arg_value = argv[i];

        if (strcmp("--rnn", arg_value) == 0) {
            printf("argument value is equal \n");
        }

        printf("argc are char array pointer %s \n", arg_value);
    }


    FILE *datei = fopen("test_data/cryptocurrency_rates_history.json", "r");
    char buffer[BUFFER_SIZE];

    if (datei != NULL) {
        printf("datei wurde geoeffnet");

        i = 0;

        real.buffer = buffer;
        int newSize;

        while ((fgets(buffer, BUFFER_SIZE, datei)) != NULL) {
            newSize = strlen(real.buffer) + strlen(buffer) + 1;

            // Allocate new buffer
            char *newBuffer = (char *) malloc(newSize);

            // do the copy and concat
            strcpy(newBuffer, real.buffer);
            strcat(newBuffer, buffer); // or strncat

            // store new pointer
            real.buffer = newBuffer;
        }

        fclose(datei);

//        printf("\n %s", real.buffer);

        

    } else {
        printf("Fehler beim Oeffnen der Datei");
    }


    return (EXIT_SUCCESS);
}

