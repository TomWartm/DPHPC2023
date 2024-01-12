#! /bin/bash
perf stat -B -e cache-references,cache-misses,cycles,instructions,branches,faults,migrations ./evaluate_trisolv_mpi
