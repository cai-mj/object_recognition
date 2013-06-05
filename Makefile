INCDIR = -I. -I./usr/local/include/opencv -I/usr/local/include
DBG    = -g
OPT    = -O3
PTH    = -pthread
CPP    = g++
CFLAGS = -c $(DBG) $(INCDIR)
LINK   = -lm -L/usr/local/lib/ -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d \
                               -lopencv_flann -lopencv_nonfree
LINKMATLAB = -L/usr/local/MATLAB/R2012a/bin/glnxa64/ -lmat -lmx

BOW: object_recognition.o
	$(CPP) -o BOW object_recognition.o $(LINK)

video_segmentation.o: object_recognition.cpp
	$(CPP) $(CFLAGS) object_recognition.cpp



clean:
	/bin/rm -f BOW *.o 

