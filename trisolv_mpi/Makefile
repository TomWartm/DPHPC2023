build : trisolv_colmaj_partialInit.cpp cluster_crossbar.xml
	smpicxx trisolv_colmaj_partialInit.cpp -O3 -DPOW=6 -o trisolv_colmaj_partialInit

run : trisolv_colmaj_partialInit
	smpirun -np 4 -platform cluster_crossbar.xml ./trisolv_colmaj_partialInit

test : roundtrip.c cluster_crossbar.xml
	smpicc -O3 roundtrip.c -o roundtrip
	smpirun -np 16 -platform cluster_crossbar.xml ./roundtrip

clean :
	rm roundtrip trisolv trisolv_colmaj_partialInit
