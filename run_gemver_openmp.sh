#!/bin/bash

# Loop over the desired thread counts
for thread in 1 2
do
    # Update the makefile to set the new NUM_THREADS value
    sed -i "s/-DNUM_THREADS=[0-9]*/-DNUM_THREADS=$thread/" Makefile

    # Recompile the program
    make clean
    make

    # Submit the job with the appropriate number of CPUs
    ./evaluate_gemver_openmp
done
