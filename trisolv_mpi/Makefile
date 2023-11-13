build : trisolv.cpp cluster_crossbar.xml
	smpicxx trisolv.cpp -O3 -fopenmp -DNDEF=32 -DOMP -o trisolv

run : trisolv
	smpirun -np 1 -platform cluster_crossbar.xml ./trisolv

test : roundtrip.c cluster_crossbar.xml
	smpicc -O3 roundtrip.c -o roundtrip
	smpirun -np 16 -platform cluster_crossbar.xml ./roundtrip

clean :
	rm roundtrip trisolv
