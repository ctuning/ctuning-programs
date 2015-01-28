#include "gcc-plugin.h"


#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "tm.h"
#include "toplev.h"
#include "gimple.h"
#include "langhooks.h"
#include "function.h"
#include "cfgloop.h"
#include "tree-pass.h"

int plugin_is_GPL_compatible;

static FILE *file1;
static char str1[30], str4[30];
static char* __restrict__ str2;

void loop_rec(struct loop *);
void print_src_info_loop(struct loop *);

static unsigned int execute_analysis_gimple(void)
{
  gimple_stmt_iterator gsi;
  gimple stmt;
  location_t loc;
  struct loop *loop, *loop1;
  const char *dname;
  loop_iterator li;
  basic_block bb;
  int bb_count=0, insns_count=0;
 
  
  dname = lang_hooks.decl_printable_name (current_function_decl, 2);
 
  if(strcmp(dname, str1)==0)
    {
      fprintf(file1, "Source file :%s\n", main_input_filename);
      fprintf(file1, "Function: %s, Start: %d, End: %d\n", dname, LOCATION_LINE(cfun->function_start_locus), LOCATION_LINE(cfun->function_end_locus));
      FOR_ALL_BB(bb)
      {
	bb_count++;
	for (gsi = gsi_start_bb (bb); !gsi_end_p (gsi); gsi_next (&gsi))
	  {
	    insns_count++;
	    stmt=gsi_stmt(gsi);
	    loc=gimple_location(stmt);
     
	    switch(gimple_code(stmt))
	      {
	      case GIMPLE_NOP:
		fprintf(file1, "NOP : ");
		break;
	      case GIMPLE_COND:
		fprintf(file1, "COND_STMT / ");
		break;
	      case GIMPLE_LABEL:
		fprintf(file1, "LABEL / ");
		break;
	      case GIMPLE_SWITCH:
		fprintf(file1, "SWITCH / ");
		break;
	      case GIMPLE_ASSIGN:
		fprintf(file1, "ASSIGN / ");
		fprintf(file1, "Number of operand %u / ", gimple_num_ops(stmt));
		switch(gimple_expr_code(stmt))
		  {
		  case PLUS_EXPR:
		    fprintf(file1, " ADD ");
		    break;
		  case MINUS_EXPR:
		    fprintf(file1, " SUB ");
		    break;
		  case MULT_EXPR:
		    fprintf(file1, " MULT ");
		    break;
		  default:
		    fprintf(file1, " SIMPLE ");
		  }
		break;
	      case GIMPLE_CALL:
		fprintf(file1, "CALL / ");
		break;
	      case GIMPLE_RETURN:
		fprintf(file1, "RETURN / ");
		break;
	      case GIMPLE_PHI:
		fprintf(file1, "PHI / ");
		break;
	      default:
		fprintf(file1, "OTHER / ");
	    
	      }
	    fprintf(file1, "SOURCE_LINE:%d\n", LOCATION_LINE(loc));
	    
	  }
      }

      loop_optimizer_init(0);
      fprintf(file1, "Number of loops in this program:%d\n", number_of_loops());
      FOR_EACH_LOOP(li, loop, 0)
	if(loop->inner)
	  {
	    fprintf(file1, "loop %d is not innermost, having children: ", loop->num);
	    loop_rec(loop->inner);
	    print_src_info_loop(loop);
	    fprintf(file1, "\n");
	  }
	else
	  {
	    fprintf(file1, "loop %d is innermost", loop->num);
	    print_src_info_loop(loop);
	    fprintf(file1, "\n");
	  }
      loop_optimizer_finalize();
      
      fprintf(file1, "Number of basic blocks: %d\n", bb_count);
      fprintf(file1, "Total insns count: %d\n", insns_count);
      fclose(file1);
    }
  return 0;
}

void print_src_info_loop(struct loop *loop)
{
  basic_block bb;
  gimple_seq seq;
  location_t loc;
  gimple stmt;

  bb=loop->header;
  seq=bb->il.gimple->seq;
  stmt=gimple_seq_first_stmt(seq);
  loc=gimple_location(stmt);
  fprintf(file1, ", Start: %d", LOCATION_LINE(loc));
  bb=loop->latch;
  seq=bb->il.gimple->seq;
  stmt=gimple_seq_first_stmt(seq);
  loc=gimple_location(stmt);
  fprintf(file1, ", End: %d", LOCATION_LINE(loc));
  fprintf(file1, ", with number of blocks contained within the loop: %d", loop->num_nodes);
}
void loop_rec(struct loop *i)
{
  fprintf(file1, "%d, ", i->num);
  if((i->inner)==NULL)
    return;

    loop_rec(i->inner);
}
static struct gimple_opt_pass pass_analysis_gimple =
{
  {
    GIMPLE_PASS,
    "analysis_gimple",                    /* name */
    NULL,                                 /* gate */
    execute_analysis_gimple,              /* execute */
    NULL,                                 /* sub */
    NULL,                                 /* next */
    0,                                    /* static_pass_number */
    0,                                    /* tv_id */
    PROP_gimple_any,                      /* properties_required */
    0,                                    /* properties_provided */
    0,                                    /* properties_destroyed */
    0,                                    /* todo_flags_start */
    TODO_dump_func                        /* todo_flags_finish */
  }
};


int plugin_init(struct plugin_name_args *plugin_info,
		struct plugin_gcc_version *version)
{
  struct register_pass_info pass_info_gimple, pass_info_rtl;
  const char *plugin_name=plugin_info->base_name;
  char *var_pass_rtl=NULL;
  char *var_pass_gimple=NULL;
  char *var_function=NULL;
  char *var_fileout=NULL;

  str2=(char* __restrict)xmalloc(1024);
  //str3=(char* __restrict)xmalloc(1024);
  
  if( (var_pass_gimple = getenv("ICI3_PASS_GIMPLE"))==NULL)
    {
      printf("Error: Environment variable ICI3_PASS_GIMPLE is not defined\n");
      exit(1);
    }
  else
    strcpy(str2, var_pass_gimple);

  if( (var_function = getenv("ICI3_FUNCTION"))==NULL)
    {
      printf("Error: Environment variable ICI3_FUNCTION is not defined\n");
      exit(1);
      }
  else
    strcpy(str1, var_function);

  if( (var_fileout = getenv("ICI3_FILEOUT"))==NULL)
    {
      printf("Error: Environment variable ICI3_FILEOUT is not defined\n");
      exit(1);
    }
  else
    strcpy(str4, var_fileout);

  file1=fopen(str4, "w");
  
  pass_info_gimple.pass = &pass_analysis_gimple.pass;
  pass_info_gimple.reference_pass_name=str2;
  pass_info_gimple.ref_pass_instance_number = 1;
  pass_info_gimple.pos_op = PASS_POS_INSERT_AFTER;
   
  register_callback(plugin_name, PLUGIN_PASS_MANAGER_SETUP, NULL, &pass_info_gimple);
  return 0;
}
