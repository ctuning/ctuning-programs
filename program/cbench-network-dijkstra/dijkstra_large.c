#include <stdio.h>
#include <stdlib.h>

#ifdef XOPENME
#include <xopenme.h>
#endif

int NUM_NODES=0;
#define NONE 9999

struct _NODE
{
  int iDist;
  int iPrev;
};
typedef struct _NODE NODE;

struct _QITEM
{
  int iNode;
  int iDist;
  int iPrev;
  struct _QITEM *qNext;
};
typedef struct _QITEM QITEM;

QITEM *qHead = NULL;

int* AdjMatrix;

int g_qCount = 0;
NODE* rgnNodes;
int ch;
int iPrev, iNode;
int i, iCost, iDist;


void print_path (NODE *rgnNodes, int chNode)
{
  if (rgnNodes[chNode].iPrev != NONE)
    {
      print_path(rgnNodes, rgnNodes[chNode].iPrev);
    }
  printf (" %d", chNode);
  fflush(stdout);
}


void enqueue (int iNode, int iDist, int iPrev)
{
  QITEM *qNew = (QITEM *) malloc(sizeof(QITEM));
  QITEM *qLast = qHead;
  
  if (!qNew) 
    {
      fprintf(stderr, "Out of memory.\n");
      exit(EXIT_FAILURE);
    }
  qNew->iNode = iNode;
  qNew->iDist = iDist;
  qNew->iPrev = iPrev;
  qNew->qNext = NULL;
  
  if (!qLast) 
    {
      qHead = qNew;
    }
  else
    {
      while (qLast->qNext) qLast = qLast->qNext;
      qLast->qNext = qNew;
    }
  g_qCount++;
  //               ASSERT(g_qCount);
}


void dequeue (int *piNode, int *piDist, int *piPrev)
{
  QITEM *qKill = qHead;
  
  if (qHead)
    {
      //                 ASSERT(g_qCount);
      *piNode = qHead->iNode;
      *piDist = qHead->iDist;
      *piPrev = qHead->iPrev;
      qHead = qHead->qNext;
      free(qKill);
      g_qCount--;
    }
}


int qcount (void)
{
  return(g_qCount);
}

void dijkstra(int chStart, int chEnd) 
{
  
  for (ch = 0; ch < NUM_NODES; ch++)
    {
      rgnNodes[ch].iDist = NONE;
      rgnNodes[ch].iPrev = NONE;
    }

  if (chStart == chEnd) 
    {
      printf("Shortest path is 0 in cost. Just stay where you are.\n");
    }
  else
    {
      rgnNodes[chStart].iDist = 0;
      rgnNodes[chStart].iPrev = NONE;
      
      enqueue (chStart, 0, NONE);
      
     while (qcount() > 0)
	{
	  dequeue (&iNode, &iDist, &iPrev);
	  for (i = 0; i < NUM_NODES; i++)
	    {
	      if ((iCost = AdjMatrix[iNode*NUM_NODES+i]) != NONE)
		{
		  if ((NONE == rgnNodes[i].iDist) || 
		      (rgnNodes[i].iDist > (iCost + iDist)))
		    {
		      rgnNodes[i].iDist = iDist + iCost;
		      rgnNodes[i].iPrev = iNode;
		      enqueue (i, iDist + iCost, iNode);
		    }
		}
	    }
	}
    }
}

int main(int argc, char *argv[]) {
  int i,j,k;
  FILE *fp;
  long ct_repeat=0;
  long ct_repeat_max=1;
  
#ifdef XOPENME
  xopenme_init(1,0);
#endif

    if (getenv("CT_REPEAT_MAIN")!=NULL) ct_repeat_max=atol(getenv("CT_REPEAT_MAIN"));

  if (argc<2) {
    fprintf(stderr, "Usage: dijkstra <filename>\n");
    fprintf(stderr, "Only supports matrix size is #define'd.\n");
    exit(EXIT_FAILURE);
  }

  /* open the adjacency matrix file */
  fp = fopen (argv[1],"r");

  fscanf(fp,"%d",&NUM_NODES);

  printf("Matrix size: %d\n", NUM_NODES);
  printf("AdjMatrix size: %d\n", (int)sizeof(int)*(NUM_NODES+1)*(NUM_NODES+1));
  printf("rgnNodesSize: %d\n", (int)sizeof(NODE)*(NUM_NODES+1));
  
  AdjMatrix=malloc(sizeof(int)*(NUM_NODES+1)*(NUM_NODES+1));
  rgnNodes=malloc(sizeof(NODE)*(NUM_NODES+1));

  /* make a fully connected matrix */
  for (i=0;i<NUM_NODES;i++) {
    for (j=0;j<NUM_NODES;j++) {
      /* make it more sparce */
      fscanf(fp,"%d",&k);
      AdjMatrix[i*NUM_NODES+j]= k;
    }
  }

#ifdef XOPENME
  xopenme_clock_start(0);
#endif

  /* finds 10 shortest paths between nodes */
  for (i=0,j=NUM_NODES/2;i<NUM_NODES;i++,j++) 
  {
    j=j%NUM_NODES;
  
    for (ct_repeat=0; ct_repeat<ct_repeat_max; ct_repeat++)
      dijkstra(i,j);

    printf("Shortest path is %d in cost. ", rgnNodes[j].iDist);
    printf("Path is: ");
    print_path(rgnNodes, j);
    printf("\n");

  }

#ifdef XOPENME
  xopenme_clock_end(0);

  xopenme_dump_state();
  xopenme_finish();
#endif

  free(AdjMatrix);
  free(rgnNodes);

  return 0;
}
