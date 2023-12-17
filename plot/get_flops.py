def get_flops_gemver(method:str, n:int) -> int:
    """
    Returns number of floating point operations (additions and multiplications) of the gemver baseline function.
    Index operations are ignored.
    """
    no_additions = 0
    no_multiplications = 0
    if method in ["baseline","gemver_mpi_2", "gemver_mpi_3","gemver_mpi_2_1", "gemver_mpi_2_new", "openmp with padding","openmp","gemver_mpi_2_new_openmp","gemver_mpi_3_new"]:
        no_additions = n*n*2 + n*n + n + n*n
        no_multiplications = n*n*2 + n*n*2 + n*n*2
    
    elif method in ["baseline blocked 1", "baseline blocked 2", "gemver_mpi_4","gemver_mpi_2_new_blocking"]:
        n = n/4

        no_additions = n*n* 32 + n*n* 16 + n*n* 16 # same as in baseline
        no_multiplications = n*n*32 + n*n*20 + n*n*20 # less as baseline
    else:
        print("Not implemented flop count for method: {}".format(method))
    
    return no_additions + no_multiplications


