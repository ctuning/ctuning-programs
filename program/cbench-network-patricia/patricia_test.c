/*
 * patricia_test.c
 *
 * Patricia trie test code.
 *
 * This code is an example of how to use the Patricia trie library for
 * doing longest-prefix matching.  We begin by adding a default
 * route/default node as the head of the Patricia trie.  This will become
 * an initialization functin (pat_init) in the future.  We then read in a
 * set of IPv4 addresses and network masks from "pat_test.txt" and insert
 * them into the Patricia trie.  I haven't _yet_ added example of searching
 * and removing nodes.
 *
 * Compiling the library:
 *     gcc -g -Wall -c patricia.c
 *     ar -r libpatricia.a patricia.o
 *     ranlib libpatricia.a
 *
 * Compiling the test code (or any other file using libpatricia):
 *     gcc -g -Wall -I. -L. -o ptest patricia_test.c -lpatricia
 *
 * Matthew Smart <mcsmart@eecs.umich.edu>
 *
 * Copyright (c) 2000
 * The Regents of the University of Michigan
 * All rights reserved
 *
 * $Id: patricia_test.c,v 1.1.1.1 2000/11/06 19:53:17 mguthaus Exp $
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* #include <endian.h> FGG commented for portability */

#include "patricia.h"

/* 
#if !defined(BYTE_ORDER) || !defined(LITTLE_ENDIAN) || !defined(BIG_ENDIAN)
#error Cannot determine endianness
#endif
*/

/*#if BYTE_ORDER == LITTLE_ENDIAN */
/* Assuming little endian architecture */
#define bswap32(x) \
     ((((x) & 0xff000000) >> 24) | (((x) & 0x00ff0000) >>  8) | \
      (((x) & 0x0000ff00) <<  8) | (((x) & 0x000000ff) << 24))
#define htonl bswap32
/* #elif BYTE_ORDER == BIG_ENDIAN */
/* Assuming big endian architecture */
/* #define htonl(x) x
#else
#error Unsupported endianness
#endif */

typedef unsigned int in_addr_t;
struct in_addr {
    in_addr_t s_addr;
};

struct MyNode {
	int foo;
	double bar;
};

int
main1(int argc, char **argv, int print)
{
	struct ptree *phead;
	struct ptree *p,*pfind;
	struct ptree_mask *pm;
	FILE *fp;
	char line[128];
	char addr_str[16];
	struct in_addr addr;
	unsigned long mask=0xffffffff;
	float time;

	if (argc<2) {
		printf("Usage: %s <TCP stream>\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	/*
	 * Open file of IP addresses and masks.
	 * Each line looks like:
	 *    10.0.3.4 0xffff0000
	 */
	if ((fp = fopen(argv[1], "r")) == NULL) {
		printf("File %s doesn't seem to exist\n",argv[1]);
		exit(EXIT_FAILURE);
	}

	/*
	 * Initialize the Patricia trie by doing the following:
	 *   1. Assign the head pointer a default route/default node
	 *   2. Give it an address of 0.0.0.0 and a mask of 0x00000000
	 *      (matches everything)
	 *   3. Set the bit position (p_b) to 0.
	 *   4. Set the number of masks to 1 (the default one).
	 *   5. Point the head's 'left' and 'right' pointers to itself.
	 * NOTE: This should go into an intialization function.
	 */
	phead = (struct ptree *)malloc(sizeof(struct ptree));
	if (!phead) {
		perror("Allocating p-trie node");
		exit(EXIT_FAILURE);
	}
	memset(phead, 0, sizeof(*phead));
	phead->p_m = (struct ptree_mask *)malloc(
			sizeof(struct ptree_mask));
	if (!phead->p_m) {
		perror("Allocating p-trie mask data");
		exit(EXIT_FAILURE);
	}
	memset(phead->p_m, 0, sizeof(*phead->p_m));
	pm = phead->p_m;
	pm->pm_data = (struct MyNode *)malloc(sizeof(struct MyNode));
	if (!pm->pm_data) {
		perror("Allocating p-trie mask's node data");
		exit(EXIT_FAILURE);
	}
	memset(pm->pm_data, 0, sizeof(*pm->pm_data));
	/*******
	 *
	 * Fill in default route/default node data here.
	 *
	 *******/
	phead->p_mlen = 1;
	phead->p_left = phead->p_right = phead;


	/*
	 * The main loop to insert nodes.
	 */
	while (fgets(line, 128, fp)) {
		/*
		 * Read in each IP address and mask and convert them to
		 * more usable formats.
		 */
		sscanf(line, "%f %d", &time, (unsigned int *)&addr);
		//inet_aton(addr_str, &addr);

		/*
		 * Create a Patricia trie node to insert.
		 */
		p = (struct ptree *)malloc(sizeof(struct ptree));
		if (!p) {
			perror("Allocating p-trie node");
			exit(EXIT_FAILURE);
		}
		memset(p, 0, sizeof(*p));

		/*
		 * Allocate the mask data.
		 */
		p->p_m = (struct ptree_mask *)malloc(
				sizeof(struct ptree_mask));
		if (!p->p_m) {
			perror("Allocating p-trie mask data");
			exit(EXIT_FAILURE);
		}
		memset(p->p_m, 0, sizeof(*p->p_m));

		/*
		 * Allocate the data for this node.
		 * Replace 'struct MyNode' with whatever you'd like.
		 */
		pm = p->p_m;
		pm->pm_data = (struct MyNode *)malloc(sizeof(struct MyNode));
		if (!pm->pm_data) {
			perror("Allocating p-trie mask's node data");
			exit(EXIT_FAILURE);
		}
		memset(pm->pm_data, 0, sizeof(*pm->pm_data));

		/*
		 * Assign a value to the IP address and mask field for this
		 * node.
		 */
		p->p_key = addr.s_addr;		/* Network-byte order */
		p->p_m->pm_mask = htonl(mask);

		pfind=pat_search(addr.s_addr,phead);
		//printf("%08x %08x %08x\n",p->p_key, addr.s_addr, p->p_m->pm_mask);
		//if(pfind->p_key==(addr.s_addr&pfind->p_m->pm_mask))
		if ((pfind->p_key==addr.s_addr) && (print==1))
		{
			printf("%f %08x: ", time, addr.s_addr);
			printf("Found.\n");
		}
		else
		{
			/*
		 	* Insert the node.
		 	* Returns the node it inserted on success, 0 on failure.
		 	*/
			//printf("%08x: ", addr.s_addr);
			//printf("Inserted.\n");
			p = pat_insert(p, phead);
		}
		if (!p) {
			fprintf(stderr, "Failed on pat_insert\n");
			exit(EXIT_FAILURE);
		}
	}

        fclose(fp);

        return 0; 
}
