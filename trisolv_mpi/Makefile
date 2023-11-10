test : roundtrip.c cluster_crossbar.xml
	smpicc -O3 roundtrip.c -o roundtrip
	smpirun -np 16 -platform cluster_crossbar.xml ./roundtrip

clean :
	rm roundtrip
