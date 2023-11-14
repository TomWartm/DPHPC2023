build : trisolv_colmaj_cp.cpp cluster_crossbar.xml
	smpicxx trisolv_colmaj_cp.cpp -O3 -POW=10 -o trisolv_colmaj_cp

run : trisolv_colmaj_cp
	smpirun -np 4 -platform cluster_crossbar.xml ./trisolv_colmaj_cp

test : roundtrip.c cluster_crossbar.xml
	smpicc -O3 roundtrip.c -o roundtrip
	smpirun -np 16 -platform cluster_crossbar.xml ./roundtrip

clean :
	rm roundtrip trisolv trisolv_colmaj
