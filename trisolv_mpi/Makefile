trisolv : trisolv.cpp cluster_crossbar.xml
	smpicxx -O3 trisolv.cpp -o trisolv
	smpirun -np 8 -platform cluster_crossbar.xml ./trisolv

test : roundtrip.c cluster_crossbar.xml
	smpicc -O3 roundtrip.c -o roundtrip
	smpirun -np 16 -platform cluster_crossbar.xml ./roundtrip

clean :
	rm roundtrip trisolv
