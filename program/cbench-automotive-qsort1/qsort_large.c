#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define UNLIMIT
#define MAXARRAY 60000 /* this number, if too large, will cause a seg. fault!! */

void qsortx(void *base, unsigned num, unsigned width,
            int (*comp)(const void *, const void *));

struct my3DVertexStruct {
  int x, y, z;
  double distance;
};

int compare(const void *elem1, const void *elem2)
{
  /* D = [(x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2]^(1/2) */
  /* sort based on distances from the origin... */

  double distance1, distance2;

  distance1 = (*((struct my3DVertexStruct *)elem1)).distance;
  distance2 = (*((struct my3DVertexStruct *)elem2)).distance;

  return (distance1 > distance2) ? 1 : ((distance1 == distance2) ? 0 : -1);
}

struct my3DVertexStruct array[MAXARRAY];

int
main1(int argc, char *argv[], int print) {
  FILE* fmisc=NULL;
  FILE *fp;
  int i,count=0;
  long x=0;
  long y=0;
  long z=0;

  if (argc<2) {
    fprintf(stderr,"Usage: qsort_large <file>\n");
    exit(EXIT_FAILURE);
  }
  else {
    fp = fopen(argv[1],"r");

    while((fscanf(fp, "%d", &x) == 1) && (fscanf(fp, "%d", &y) == 1) && (fscanf(fp, "%d", &z) == 1) &&  (count < MAXARRAY)) {
	 array[count].x = x;
	 array[count].y = y;
	 array[count].z = z;
	 array[count].distance = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
	 count++;
    }

    fclose(fp);
  }
  if (print==1) {
      printf("\nSorting %d vectors based on distance from the origin.\n\n",
             count);
  }
  qsortx(array,count,sizeof(struct my3DVertexStruct),compare);

  if (print==1) {
      if ((fmisc=fopen("tmp-output.tmp","wt"))==NULL)
          {
              fprintf(stderr,"\nError: Can't open output file\n");
              exit(EXIT_FAILURE);
          }
   for(i=0;i<count;i+=count/100)
     fprintf(fmisc, "%d %d %d\n", array[i].x, array[i].y, array[i].z);
  
   fclose(fmisc);
  }
  return 0;
}
