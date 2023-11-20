smpi-build-partialInit : trisolv_colmaj_partialInit.cpp cluster_crossbar.xml
	smpicxx trisolv_colmaj_partialInit.cpp -O3 -DPOW=6 -o trisolv_colmaj_partialInit.ex

smpi-run-partialInit : trisolv_colmaj_partialInit.ex
	smpirun -np 4 -platform cluster_crossbar.xml ./trisolv_colmaj_partialInit.ex
	
euler-build-partialInit : trisolv_colmaj_partialInit.cpp
	mpicxx trisolv_colmaj_partialInit.cpp -O3 -DPOW=12 -o trisolv_colmaj_partialInit.ex

euler-run-partialInit : trisolv_colmaj_partialInit.ex
	module load gcc openmpi
	sbatch -C ib --ntasks=48 --wrap="mpirun -np 32 ./trisolv_colmaj_partialInit.ex"

test : roundtrip.c cluster_crossbar.xml
	smpicc -O3 roundtrip.c -o roundtrip.ex
	smpirun -np 16 -platform cluster_crossbar.xml ./roundtrip.ex

clean :
	rm *.ex
