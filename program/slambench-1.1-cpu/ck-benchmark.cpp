/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#include <kernels.h>
#include <interface.h>
#include <stdint.h>
#include <vector>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <csignal>

#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>
#include <iomanip>

inline double tock() {
	return 0;
}	



/***
 * This program loop over a scene recording
 */

int main(int argc, char ** argv) {
	DepthReader * reader;


	std::cout.precision(10);
	std::cerr.precision(10);

	free(depthRender);
	return 0;
}
