TARGETS = getBoxLocation clusterTrajectory convert merge meanIntensity threshold imageContrast visualizeFeatures
CC = g++
CFLAGS=`pkg-config opencv --cflags` -std=c++0x -g -Wall
LIBS=`pkg-config opencv --libs`

all:$(TARGETS)

getBoxLocation: getBoxLocation.cpp
	$(CC) $(CFLAGS) getBoxLocation.cpp -o getBoxLocation $(LIBS)
	rm -rf *.dSYM

merge: merge.cpp
	$(CC) $(CFLAGS) merge.cpp -o merge $(LIBS)
	rm -rf *.dSYM

convert: convert.cpp
	$(CC) $(CFLAGS) convert.cpp -o convert $(LIBS)
	rm -rf *.dSYM

ViBe.o: ../bg_test/ViBe/ViBe.h ../bg_test/ViBe/ViBe.cpp
	$(CC) -c $(CFLAGS) ../bg_test/ViBe/ViBe.cpp

clusterTrajectory: clusterTrajectory.cpp ViBe.o
	$(CC) $(CFLAGS) clusterTrajectory.cpp ViBe.o -o clusterTrajectory $(LIBS)

meanIntensity: meanIntensity.cpp 
	$(CC) $(CFLAGS) meanIntensity.cpp -o meanIntensity $(LIBS)
	rm -rf *.dSYM

threshold: threshold.cpp 
	$(CC) $(CFLAGS) threshold.cpp -o threshold $(LIBS)
	rm -rf *.dSYM

imageContrast: contrast.cpp
	$(CC) $(CFLAGS) contrast.cpp -o imageContrast $(LIBS)
	rm -rf *.dSYM
visualizeFeatures: visualizeFeatures.cpp
	$(CC) $(CFLAGS) visualizeFeatures.cpp -o visualizeFeatures $(LIBS)

clean:
	rm -rf $(TARGETS) *.o *.dSYM
