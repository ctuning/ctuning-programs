/* NIST Secure Hash Algorithm */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "sha.h"

int main1(int argc, char **argv, int print)
{
    FILE *fin;
    SHA_INFO sha_info;

    if (argc < 2) {
	fin = stdin;
        sha_stream(&sha_info, fin);
        if (print)
            sha_print(&sha_info);
    } else {
	while (--argc) {
	    fin = fopen(*(++argv), "rb");
	    if (fin == NULL) {
		fprintf(stderr, "error opening %s for reading\n", *argv);
                exit(EXIT_FAILURE);
	    } else {
                sha_stream(&sha_info, fin);
		if (print)
                    sha_print(&sha_info);
		fclose(fin);
	    }
	}
    }
    return(0);
}
