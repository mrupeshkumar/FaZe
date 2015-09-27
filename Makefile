# Makefile for Project Timeline
#
# Your compiler
CXX = g++

# Compilation flags
# '-g' turns debugging flags on.
# Not Using O2 flag for optimisation.
CXXFLAGS = -g -I./include -I./src/dlib/all/source.cpp -ljpeg -mavx -lm -lpthread -lX11 -DDLIB_HAVE_BLAS -DNDEBUG  -DDLIB_JPEG_SUPPORT -DDLIB_HAVE_AVX  -O3 `pkg-config --cflags opencv`

# Linker flags
# When you need to add a library
LDFLAGS = -ljpeg -mavx -lm -lpthread -lX11 `pkg-config --libs opencv` -DDLIB_HAVE_BLAS -DNDEBUG  -DDLIB_JPEG_SUPPORT -DDLIB_HAVE_AVX  -O3

# all is a target
# $(VAR) gives value of the variable.
# $@ stores the target
# $^ stores the dependency
# all: bin/oic bin/facegesmatch bin/facegescreate bin/facegeslisten

bin/faze: obj/dlib.o obj/fazeModel.o obj/pupilDetectionCDF.o obj/pupilDetectionSP.o obj/util.o obj/main.o
	$(CXX) -o $@ $^ $(LDFLAGS)

obj/dlib.o: src/dlib/all/source.cpp
	mkdir -p obj bin
	$(CXX) -c $(CXXFLAGS) -o $@ $<

obj/fazeModel.o: src/fazeModel.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

obj/pupilDetectionCDF.o: src/pupilDetectionCDF.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

obj/pupilDetectionSP.o: src/pupilDetectionSP.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

obj/util.o: src/util.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

#obj/gazeComputationQE.o: src/gazeComputationQE.cpp
#	$(CXX) -c $(CXXFLAGS) -o $@ $<

#obj/gazeComputationVA.o: src/gazeComputationVA.cpp
#	$(CXX) -c $(CXXFLAGS) -o $@ $<

obj/main.o: src/main.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

# .PHONY tells make that 'all' or 'clean' aren't _actually_ files, and always
# execute the compilation action when 'make all' or 'make clean' are used
.PHONY: all oic

# Delete all the temporary files we've created so far
clean:
	mv obj/dlib.o .
	rm -rf obj/*
	mv dlib.o obj/
	rm -rf bin/*