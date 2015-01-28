#include "gcc-plugin.h"


#include "config.h"
#include "system.h"
#include "coretypes.h"
#include "tm.h"
#include "toplev.h"
#include "rtl.h"
#include "gimple.h"
#include "langhooks.h"
//#include "tm_p.h"
//#include "hard-reg-set.h"
//#include "regs.h"
//#include "function.h"
//#include "flags.h"
//#include "insn-config.h"
//#include "insn-attr.h"
//#include "except.h"
//#include "toplev.h"
//#include "recog.h"
//#include "sched-int.h"
//#include "target.h"
//#include "cfglayout.h"
#include "cfgloop.h"
//#include "cfghooks.h"
//#include "expr.h"
//#include "params.h"
//#include "gcov-io.h"
//#include "ddg.h"
//#include "timevar.h"
#include "tree-pass.h"


static FILE *file1;
static char str1[30], str4[30];
static char* __restrict__ str2;
static char* __restrict__ str3;

int plugin_is_GPL_compatible;

static bool mem_ref_p;

static int mark_mem_use (rtx *x, void *data ATTRIBUTE_UNUSED)
{
  if (MEM_P (*x))
    mem_ref_p = true;
  return 0;
}

static void mark_mem_use_1 (rtx *x, void *data)
{
  for_each_rtx (x, mark_mem_use, data);
}

static bool mem_read_insn_p (rtx insn)
{
  mem_ref_p = false;
  note_uses (&PATTERN (insn), mark_mem_use_1, NULL);
  return mem_ref_p;
}

static void mark_mem_store (rtx loc, const_rtx setter ATTRIBUTE_UNUSED, void *data ATTRIBUTE_UNUSED)
{
  if (MEM_P (loc))
    mem_ref_p = true;
}

static bool mem_write_insn_p (rtx insn)
{
  mem_ref_p = false;
  note_stores (PATTERN (insn), mark_mem_store, NULL);
  return mem_ref_p;
}

static bool rtx_mem_access_p (rtx x)
{
  int i, j;
  const char *fmt;
  enum rtx_code code;

  if (x == 0)
    return false;

  if (MEM_P (x))
    return true;

  code = GET_CODE (x);
  fmt = GET_RTX_FORMAT (code);
  for (i = GET_RTX_LENGTH (code) - 1; i >= 0; i--)
    {
      if (fmt[i] == 'e')
        {
          if (rtx_mem_access_p (XEXP (x, i)))
            return true;
        }
      else if (fmt[i] == 'E')
        for (j = 0; j < XVECLEN (x, i); j++)
          {
            if (rtx_mem_access_p (XVECEXP (x, i, j)))
              return true;
          }
    }
  return false;
}

static bool mem_access_insn_p (rtx insn)
{
  return rtx_mem_access_p (PATTERN (insn));
}

static unsigned int execute_analysis_rtl(void)
{
  rtx first_insn, other_insn;
  int i, j=0, code_labels_count=0, *labels=NULL, *insn_after_label=NULL, flag=0, k, where=0;
  const char *dname;
  RTX_CODE code;
  struct loop *loop;
  location_t loc;
  int num_arith=0, num_loops=0, num_insns=0, num_loads=0, num_stores=0, num_cond_jmps=0, num_uncond_jmps=0;
  
  labels=xmalloc(sizeof(int));
  dname = lang_hooks.decl_printable_name (current_function_decl, 2);
  if(strcmp(dname, str1)==0)
    {
      i=get_max_insn_count();

      first_insn = crtl->emit.x_first_insn;
      do	
	{
	  code = GET_CODE(first_insn);
	  switch(GET_RTX_CLASS(code))
	    {
	    case RTX_EXTRA://First case of RTX_CLASS
		fprintf(file1, "Instruction %d is RTX_EXTRA", XINT(first_insn, 0));
		switch(code)
		  {
		  case CODE_LABEL:
		    labels[code_labels_count]=XINT(first_insn, 0);
		    code_labels_count++;
		    labels=(int *)xrealloc(labels, (code_labels_count+1)*sizeof(int *));
		    insn_after_label=(int *)xrealloc(insn_after_label, (code_labels_count+1)*sizeof(int *));
		    insn_after_label[code_labels_count-1]=0;
		    break;
		  }//switch within RTX_EXTRA ends here
		break;
	    case RTX_INSN://Second case of RTX_CLASS
	      fprintf(file1, "Instruction %d is RTX_INSN /", XINT(first_insn, 0));

	      switch(code)
		{
		case JUMP_INSN:
		  flag=0;
		  where=0;
		  other_insn=XEXP(first_insn, 4);
		  other_insn=XEXP(other_insn, 1);
		  if(GET_CODE(other_insn)==LABEL_REF)
		    {
		      fprintf(file1, "Unconditional jump");
		      num_uncond_jmps++;
		      other_insn=XEXP(other_insn, 0);	
		      where=XINT(other_insn, 0);
		      for(k=0; k<code_labels_count; k++)
			{
			  if(where==labels[k]) flag=1;
			}
		    }
		   else
		    {
		      if(GET_CODE(other_insn)==IF_THEN_ELSE)
		    	{
			  other_insn=XEXP(other_insn, 1);
			  if(GET_CODE(other_insn)==LABEL_REF)
			    {
			      other_insn=XEXP(other_insn, 0);
			      where=XINT(other_insn, 0);
			      num_cond_jmps++;
			      for(k=0; k<code_labels_count; k++)
				{
				  if(where==labels[k]) flag=1;
				}
			      
			      if(flag==1)
				{	
				  fprintf(file1, " LOOP /");
				  
				  loop_optimizer_init(LOOPS_HAVE_PREHEADERS | LOOPS_HAVE_SIMPLE_LATCHES);
				  num_loops=number_of_loops();
				  loop=get_loop(num_loops-1);
				  other_insn=loop->header->il.rtl->head_;
				  other_insn=PREV_INSN(other_insn);
				  other_insn=PREV_INSN(other_insn);
				  loc =  RTL_LOCATION(other_insn);
				  fprintf(file1, " last_insn : %d /", expand_location(loc).line);
				  other_insn=loop->latch->il.rtl->head_;
				  other_insn=NEXT_INSN(other_insn);
				  other_insn=NEXT_INSN(other_insn);
				  loc =  RTL_LOCATION(other_insn);
				  fprintf(file1, " first_insn: %d /", expand_location(loc).line);
				  fprintf(file1, " n_insn: %d", loop->av_ninsns);
				  loop_optimizer_finalize();
				}
			    }
			}
		      
			}
		  break;
		default:
		  ;
		  }//switch within RTX_INSN ends here
	      if(mem_read_insn_p(first_insn))
		{
		  loc=RTL_LOCATION(first_insn);
		  fprintf(file1, "LOAD / SRC_LINE: %d ", expand_location(loc).line );
		  num_loads++;
		}
	      if(mem_write_insn_p(first_insn))
		{
		  loc=RTL_LOCATION(first_insn);
		  fprintf(file1, "STORE / SRC_LINE: %d", expand_location(loc).line);
		  num_stores++;
		}
	      
	      other_insn=XEXP(first_insn, 4);
	      if(GET_CODE(other_insn)==PARALLEL)
		{
		  other_insn=XVECEXP(other_insn, 0, 0);
		  other_insn=XEXP(other_insn, 1);
		  switch(GET_RTX_CLASS(GET_CODE(other_insn)))
		    {
		    case RTX_COMM_ARITH:
		    case RTX_BIN_ARITH:
		      fprintf(file1, "ARITH");
		      num_arith++;
		      break;
		    }
		}
	    default:
	      ;
	    }
	  j++;
	  first_insn=NEXT_INSN(first_insn);
	  fprintf(file1, "\n");
	}
      while(j<i);
      fprintf(file1, "Number of STORE instructions: %d\n", num_stores);
      fprintf(file1, "Number of LOAD instructions: %d\n", num_loads);
      fprintf(file1, "Number of loops: %d\n", num_loops);
      fprintf(file1, "Number of Arithmetic instructions: %d\n", num_arith);
      fprintf(file1, "Total number of RTL instructions: %d\n", i);
      fclose(file1);
    }
   return 0;
}

static struct rtl_opt_pass pass_analysis_rtl = 
  {
    {
      RTL_PASS,
      "analysis_rtl",
    NULL,
    execute_analysis_rtl, 
    NULL,
    NULL,
    0,
    0,
    PROP_rtl,
    0,
    0,
    0,
    TODO_dump_func
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
  str3=(char* __restrict)xmalloc(1024);
  
  if( (var_pass_gimple = getenv("ICI3_PASS_GIMPLE"))==NULL)
    {
      printf("Error: Environment variable ICI3_PASS_GIMPLE is not defined\n");
      exit(1);
    }
  else
    strcpy(str2, var_pass_gimple);

  if( (var_pass_rtl = getenv("ICI3_PASS_RTL"))==NULL)
    {
      printf("Error: Environmen variable ICI3_PASS_RTL is not defined\n");
      exit(1);
    }
  else
    strcpy(str3, var_pass_rtl);

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

  pass_info_rtl.pass = &pass_analysis_rtl.pass;
  pass_info_rtl.reference_pass_name=str3;
  pass_info_rtl.ref_pass_instance_number = 1;
  pass_info_rtl.pos_op = PASS_POS_INSERT_AFTER;
  
  register_callback(plugin_name, PLUGIN_PASS_MANAGER_SETUP, NULL, &pass_info_rtl);
  return 0;
}
