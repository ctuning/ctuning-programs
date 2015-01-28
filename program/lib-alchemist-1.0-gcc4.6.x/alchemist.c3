/*

 Alchemist low-level plugin for GCC

 Author(s): Abdul Wahid Memon
            Grigori Fursin (http://cTuning.org/lab/people/gfursin)

 Started on 2011.09
*/

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
#include "tree.h"
#include "tree-pass.h"
#include "tree-flow.h"
#include "tree-inline.h"
#include "tree-flow-inline.h"
#include "version.h"

#include <openme.h>
#include <cJSON.h>

#include "alchemist.h"

int plugin_is_GPL_compatible;

/* Alchemist input file environment */
static char *alc_env="CM_ALCHEMIST_INPUT_FILE";

/* Alchemist ini file */
static cJSON *j_ini=NULL;

/* Alchemist aggregated output */
static char output_file[1024]="";
static cJSON *j_out=NULL;

/* Alchemist various buffers and temp vars*/
static char *str=NULL;
static char buf[1024];

static cJSON *j_tmp=NULL; 
static cJSON *j_tmp1=NULL; 
static cJSON *j_tmp2=NULL; 











static char func_name[255], **funcs_per_file = NULL, input_file[255];
static int func_count = 0;
static struct opt_pass *list_temp;

static bool mem_ref_p;

/* Auxiliary function for mem_read_insn_p.  */
static int
mark_mem_use (rtx *x, void *data ATTRIBUTE_UNUSED)
{
  if (MEM_P (*x))
    mem_ref_p = true;
  return 0;
}

/* Auxiliary function for mem_read_insn_p.  */
static void
mark_mem_use_1 (rtx *x, void *data)
{
  for_each_rtx (x, mark_mem_use, data);
}

/* Returns nonzero if INSN reads from memory.  */
static bool
mem_read_insn_p (rtx insn)
{
  mem_ref_p = false;
  note_uses (&PATTERN (insn), mark_mem_use_1, NULL);
  return mem_ref_p;
}

static void
mark_mem_store (rtx loc, const_rtx setter ATTRIBUTE_UNUSED, void *data ATTRIBUTE_UNUSED)
{
  if (MEM_P (loc))
    mem_ref_p = true;
}

/* Returns nonzero if INSN writes to memory.  */
static bool
mem_write_insn_p (rtx insn)
{
  mem_ref_p = false;
  note_stores (PATTERN (insn), mark_mem_store, NULL);
  return mem_ref_p;
}

/* Returns nonzero if X has access to memory.  */
static bool
rtx_mem_access_p (rtx x)
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

/* Returns nonzero if INSN reads to or writes from memory.  */
static bool
mem_access_insn_p (rtx insn)
{
  return rtx_mem_access_p (PATTERN (insn));
}


cJSON *load_file(char *file_name);
struct opt_pass * retreive_pass(char *pass, struct opt_pass **pass_list);
bool search_pass_list(struct opt_pass *pass_tmp, struct opt_pass **pass_list);
char * search_pass_list_l(struct opt_pass *pass);
bool add_to_json(const char *func_name, char *pass_list_name, struct opt_pass *pass, int flag);
void add_to_json_ipa(struct opt_pass *pass);
char * pass_type(int pass_type);
void loop_rec(struct loop *loop, int *childs);
void loop_rec1(struct loop *loop, int *childs);

static unsigned int execute_update_memref(void *gcc_data, void *user_data)
{
  const char *dname;
  basic_block bb;
  gimple_stmt_iterator gsi;
  gimple stmt;
  tree lhs, rhs, var_decl, var_decl2, ssa_name, ssa_name1;
  int block_count=1;

  dname = lang_hooks.decl_printable_name (current_function_decl, 2);
  
  printf("TARGET_MEM_REF: %d, MEM_REF: %d\n", TARGET_MEM_REF, MEM_REF);
  
  var_decl=build_decl(UNKNOWN_LOCATION, VAR_DECL, get_identifier("a"), float_type_node);
  DECL_INITIAL(var_decl)=build_real_from_int_cst(float_type_node, build_int_cst(integer_type_node, 0));
  add_local_decl(cfun, var_decl);

  var_decl2 = build_decl(UNKNOWN_LOCATION, VAR_DECL, get_identifier("a_ptr"), float_ptr_type_node);
  add_local_decl(cfun, var_decl2);

  tree addr_of_var = build_fold_addr_expr(var_decl);

  FOR_EACH_BB(bb)
  {
    block_count++;
    for(gsi=gsi_start_bb(bb); !gsi_end_p(gsi); gsi_next(&gsi))
      {
	stmt=gsi_stmt(gsi);
	switch(gimple_code(stmt))
	  {
	  case GIMPLE_ASSIGN:
	    lhs=gimple_get_lhs(stmt);
	    rhs=gimple_assign_rhs1(stmt);

	    tree tmr_new;
	    basic_block bb = ENTRY_BLOCK_PTR->next_bb;
	    gimple_stmt_iterator gsi1 = gsi_start_bb(bb);
	    gimple stmt2;
	    
	    //printf("block: %d, code_lhs: %d, code_rhs: %d\n", block_count, TREE_CODE(lhs), TREE_CODE(rhs));
	    
	    if(TREE_CODE(lhs) == TARGET_MEM_REF)
	      {
			
		ssa_name1 = make_ssa_name(var_decl2, NULL);
		tree addr_of_var = build_fold_addr_expr(var_decl);
		stmt2 = gimple_build_assign(ssa_name1, addr_of_var);
		SSA_NAME_DEF_STMT(ssa_name1) = stmt2;
		gsi_insert_before(&gsi, stmt2, GSI_CONTINUE_LINKING);
		gsi_next(&gsi);
		
		tmr_new = build5(TARGET_MEM_REF, TREE_TYPE(lhs), ssa_name1, TMR_OFFSET(lhs), NULL, NULL, NULL);
		gimple_set_lhs(stmt, tmr_new);
		
		//printf("mem_ref updated\n");
		
	      }
	     if(TREE_CODE(rhs) == TARGET_MEM_REF)
	       {
		       	
		 ssa_name1 = make_ssa_name(var_decl2, NULL);
		 tree addr_of_var1 = build_fold_addr_expr(var_decl);
		 stmt2 = gimple_build_assign(ssa_name1, addr_of_var1);
		 SSA_NAME_DEF_STMT(ssa_name1) = stmt2;
		 gsi_insert_before(&gsi, stmt2, GSI_CONTINUE_LINKING);
		 gsi_next(&gsi);
		 
		 tmr_new = build5(TARGET_MEM_REF, TREE_TYPE(rhs), ssa_name1, TMR_OFFSET(rhs), NULL, NULL, NULL);
		 gimple_assign_set_rhs1(stmt, tmr_new);
		 
		 //printf("mem_ref updated\n");
			
	       }

	  }
      }
  }
  printf("All Target memory references updated.\n");
  return 0;
}

void record_code_structure(void *gcc_data, void *user_data)
{
  const char *dname;
  basic_block *bb;
  loop_iterator li;
  struct loop *loop;
  RTX_CODE code;
  rtx insn, other_insn;
  location_t loc;

  int *p = NULL, *childs = NULL, **children = NULL, num_loops=0;
  int i, j, k, count_outer=0, label_count=0, *labels=NULL, *label_bb=NULL, where;
  cJSON *json_program, *json_func, *json_loop, *json_tmp, *json_tmp1, *json_str, *json_bb, *j_inisn, *json_array_insns, *json_array_bb, *json_array_loops;
  char str[10], str1[10];

  dname = lang_hooks.decl_printable_name (current_function_decl, 2);
  loop_optimizer_init(LOOPS_NORMAL);

  num_loops = number_of_loops();
  if(num_loops == 1)
    {
      loop_optimizer_finalize();
      return;
    }

  printf("Number of loops in function %s : %d\n", dname, number_of_loops());
      
  FOR_EACH_LOOP(li, loop, 0)
    {
      if(VEC_length(loop_p, loop->superloops)==1)
	{
	  p = (int *)xrealloc(p, (count_outer+1)*sizeof(int *));
	  p[count_outer]=loop->num;
		  
	  childs = (int *)xrealloc(childs, (count_outer+1)*sizeof(int *));
	  childs[count_outer]=0;
		  
	  if(loop->inner!=NULL)
	    loop_rec(loop->inner, &childs[count_outer]);
	  count_outer++;
	}  
    }
    
  children=(int **)xmalloc(count_outer*sizeof(int *));
  for(i=0; i<count_outer; i++)
    {
      loop=get_loop(p[i]);
      if(childs[i]==0)
	{
	  children[i]=(int *)xmalloc(sizeof(int));
	  children[i][0]=0;
	  continue;
	}
      else
	{
	  children[i]=(int *)xmalloc(childs[i] * sizeof(int));
	  loop_rec1(loop->inner, children[i]);
	}
    }
      
  // Collect all labels
  labels=xmalloc(sizeof(int));
  label_bb=xmalloc(sizeof(int));

  /*
  insn=crtl->emit.x_first_insn;
  i=get_max_insn_count;
  j=0;
  do
    {
      code=GET_CODE(insn);
      switch(GET_RTX_CLASS(code))
	{
	case RTX_EXTRA:
	  switch(code)
	    {
	      case CODE_LABEL:
		printf("%d\n", XINT(insn, 0));
	    }
	}
      j++;
      insn=NEXT_INSN(insn);
    }while(j<i);
  */
  json_func=cJSON_CreateObject();
  json_str=cJSON_CreateString("function");
  cJSON_AddItemToObject(json_func, "type", json_str);
  json_str=cJSON_CreateString(dname);
  cJSON_AddItemToObject(json_func, "function_name", json_str);
  json_str=cJSON_CreateString(main_input_filename);
  cJSON_AddItemToObject(json_func, "source_filename", json_str);
  sprintf(str, "%d", LOCATION_LINE(cfun->function_start_locus));
  json_str=cJSON_CreateString(str);
  cJSON_AddItemToObject(json_func, "start_line", json_str);
  sprintf(str, "%d", LOCATION_LINE(cfun->function_end_locus));
  json_str=cJSON_CreateString(str);
  cJSON_AddItemToObject(json_func, "stop_line", json_str);

  json_array_loops=cJSON_CreateArray();

  for(i=0; i<count_outer; i++)
    {
      loop=get_loop(p[i]);
      bb=get_loop_body(loop);

      json_loop = cJSON_CreateObject();
      json_str = cJSON_CreateString("loop");
      cJSON_AddItemToObject(json_loop, "type", json_str);
      sprintf(str, "%d", loop->num);
      json_str = cJSON_CreateString(str);
      cJSON_AddItemToObject(json_loop, "index", json_str);
      
      insn=bb[0]->il.rtl->end_;
      loc=RTL_LOCATION(insn);
      sprintf(str, "%d", expand_location(loc).line);
      json_str=cJSON_CreateString(str);
      cJSON_AddItemToObject(json_loop, "start_line", json_str);
      
      insn=bb[loop->num_nodes-1]->il.rtl->end_;
      insn=RTX_PREV(insn);
      loc=RTL_LOCATION(insn);
      sprintf(str, "%d", expand_location(loc).line);
      json_str=cJSON_CreateString(str);
      cJSON_AddItemToObject(json_loop, "end_line", json_str);

      printf("estimated: %d", loop->nb_iterations_upper_bound);
      // Basic blocks array
      json_array_bb=cJSON_CreateArray();
      printf("%d contains %d bbs\n", loop->num, loop->num_nodes);

      for(j=0; j<loop->num_nodes; j++)
	{
	  json_bb = cJSON_CreateObject();
	  json_str = cJSON_CreateString("bb");
	  cJSON_AddItemToObject(json_bb, "type", json_str);
	  sprintf(str, "%d", bb[j]->index);
	  json_str = cJSON_CreateString(str);
	  cJSON_AddItemToObject(json_bb, "index", json_str);

	  json_array_insns=cJSON_CreateArray();	  
	  FOR_BB_INSNS(bb[j], insn)
	    {
	      /*
	      printf("bb: %d\n", bb[j]->index);
	      loc=RTL_LOCATION(insn);
	      printf("%d\n", expand_location(loc).line);
	      */
	      code=GET_CODE(insn);
	      switch(GET_RTX_CLASS(code))
		{
		case RTX_EXTRA:
		  continue;
		case RTX_INSN:
		  strcpy(str, "");
		  
		  switch(code)
		    {
		    case JUMP_INSN:
		      strcpy(str, "jump_insn");
		      where=0;
		      other_insn=XEXP(insn, 4);
		      other_insn=XEXP(other_insn, 1);
		      if(GET_CODE(other_insn) == LABEL_REF)
			{
			  other_insn=XEXP(other_insn, 0);
			  where=XINT(other_insn, 0);
			  printf("printing all labels\n");
			  for(k=0; k<label_count; k++)
			    printf("%d %d\n", labels[k], label_bb[k]);
			  for(k=0; k<label_count; k++)
			    {
			      if(where == labels[k])
				{
				  sprintf(str1, "_%d", label_bb[k]);
				  strcat(str, str1);
				  break;
				}
			    }
			}
		      else
			{
			  if(GET_CODE(other_insn) == IF_THEN_ELSE)
			    {
			      other_insn=XEXP(other_insn, 1);
			      if(GET_CODE(other_insn) == LABEL_REF)
				{
				  other_insn=XEXP(other_insn, 0);
				  where=XINT(other_insn, 0);
				  for(k=0; k<label_count; k++)
				    {
				      if(where == labels[k])
					{
					  sprintf(str1, "_%d", label_bb[k]);
					  strcat(str, str1);
					  break;
					}
				    }
				}
			    }
			}
						  
		      //continue;
		    }
		  other_insn=XEXP(insn, 4);
		  if(GET_CODE(other_insn) == PARALLEL)
		    {
		      
		      other_insn=XVECEXP(other_insn, 0, 0);
		      other_insn=XEXP(other_insn, 1);
		      switch(GET_RTX_CLASS(GET_CODE(other_insn)))
			{
			case RTX_COMM_ARITH:
			case RTX_BIN_ARITH:
			  strcat(str, "arithmetic");
			  if(mem_read_insn_p(insn))
			    strcat(str, "_load");
			  if(mem_write_insn_p(insn))
			    strcat(str, "_store");
			}
		      break;
		    }
		  if(mem_read_insn_p(insn) && mem_write_insn_p(insn))
		    {
		      strcpy(str, "load/store");
		      break;
		    }
		  if(mem_write_insn_p(insn))
		    strcat(str, "store");
		  if(mem_read_insn_p(insn))
		    strcat(str, "load");
		  break;		
		default:
		  printf("No match\n");
		}
	      j_inisn=cJSON_CreateObject();
	      json_str=cJSON_CreateString(str);
	      cJSON_AddItemToObject(j_inisn, "insn_type", json_str);
	      cJSON_AddItemToArray(json_array_insns, j_inisn);
	    }
	   cJSON_AddItemToObject(json_bb, "array", json_array_insns);
	   cJSON_AddItemToArray(json_array_bb, json_bb);
	}
      cJSON_AddItemToObject(json_loop, "array", json_array_bb);
      cJSON_AddItemToArray(json_array_loops, json_loop);
      free(bb);
    }
  cJSON_AddItemToObject(json_func, "array",  json_array_loops);
  json_program=cJSON_GetObjectItem(j_out, "#program#");
  cJSON_AddItemToArray(json_program, json_func);
  loop_optimizer_finalize();
}

// For the moment, just play with all_passes list
void substitute_passes(void *gcc_data, void *user_data)
{
 cJSON *j_ini, *json_tmp, *json_attr, *json_child;
  int list_size;
  const char *dname;
  struct opt_pass *pass_tmp;

  if(current_function_decl != NULL)
    {
       dname = lang_hooks.decl_printable_name (current_function_decl, 2);
       if(strcmp(dname, func_name) == 0)
	 {
	   printf("Substituting passes\n");
	   list_temp = all_passes;
	   
	   //read passes from input_file
	   j_ini = load_file(input_file);
	   json_tmp = cJSON_GetObjectItem(j_ini, func_name);
	   json_tmp = cJSON_GetObjectItem(json_tmp, "all_passes");         // all_passes in an array of objects
	   list_size = cJSON_GetArraySize(json_tmp);
	   json_tmp = json_tmp->child;
	   
	   while(list_size > 1)
	     {
	       json_child = cJSON_GetObjectItem(json_tmp, "pass");
	       //json_attr = cJSON_GetObjectItem(json_tmp, "attr");
	       
	       //if(strcmp(json_attr->valuestring, "save") == 0)
	       //{
		   // retrieve pass from all_passes
	       pass_tmp = retreive_pass(json_child->valuestring, &all_passes);
	       if(pass_tmp)
		 execute_one_pass(pass_tmp);
	       //}
	       json_tmp = json_tmp->next;
	       list_size--;
	     }
	   all_passes = retreive_pass("*clean_state", &all_passes);
	 }
    }
}

void test1(void *gcc_data, void *user_data)
{
  const char *dname;
  
  if(current_function_decl != NULL)
    {
      dname = lang_hooks.decl_printable_name (current_function_decl, 2);

       if(strcmp(dname, func_name) == 0)
	{
	  all_passes = list_temp;
	}
    }
}
void test(void *gcc_data, void *user_data)
{
  const char *dname;

  if(current_function_decl != NULL)
    {
      dname = lang_hooks.decl_printable_name (current_function_decl, 2);

      if(strcmp(dname, func_name) == 0)
	printf("%s\n", current_pass->name);
    }
}

void save_executed_passes(void *gcc_data, void *user_data)
{
  const char *dname;
  cJSON *json_tmp, *json_str;
  int flag1 = 0, i;
  bool success=false;
  
  if(current_function_decl == NULL)
    add_to_json_ipa(current_pass);
  else
    {
      dname = lang_hooks.decl_printable_name (current_function_decl, 2);
      
      if(strcmp(dname, func_name) != 0)
	{
	  strcpy(func_name, dname);
	  
	  for(i=0; i<func_count; i++)
	    {
	      //locate if function already exists
	      if(strcmp(funcs_per_file[i], func_name)==0)
		{
		  flag1=1;
		  break;
		}
	    }
	  if(flag1 == 0)            // function not found in functions array
	    {
	      funcs_per_file=(char **)xrealloc(funcs_per_file, (func_count+1)*sizeof(char *));
	      funcs_per_file[func_count++]=xstrdup(func_name);
	      // This is a new function so we have to setup object and array for the function (only done once for each function
	      add_to_json(dname, search_pass_list_l(current_pass), current_pass, 1);
	      flag1=0;                // reset it for any new function
	    }
	  else
	    add_to_json(dname, search_pass_list_l(current_pass), current_pass, 0);
	}
      else
	add_to_json(dname, search_pass_list_l(current_pass), current_pass, 0);
    }
  return;
}


static struct gimple_opt_pass pass_update_memref = {
  {
    .type = GIMPLE_PASS,
    .name = "remove_stmt",                     
    .gate = NULL,                            
    .execute = execute_update_memref,                 
    .sub = NULL,                              
    .next = NULL,                              
    .static_pass_number = 0,                                   
    .tv_id = TV_NONE,                                    
    .properties_required = PROP_gimple_any,                      
    .properties_provided = 0,                                   
    .properties_destroyed = 0,                                   
    .todo_flags_start = 0,                                 
    .todo_flags_finish = 0                        
  }
};

static struct rtl_opt_pass pass_extract_loops =
{
  {
    .type = RTL_PASS,
    .name = "extract_loops",	        	  
    .gate = NULL,               		  
    .execute = record_code_structure,                
    .sub = NULL,                              
    .next = NULL,                             
    .static_pass_number = 0,                                    
    .tv_id = TV_LOOP,                                  
    .properties_required = 0,                    
    .properties_provided = 0,                                   
    .properties_destroyed = 0,                                    
    .todo_flags_start = 0,                                    
    .todo_flags_finish = 0                        
  }
};


struct opt_pass * retreive_pass(char *pass_name, struct opt_pass **pass_list)
{
  struct opt_pass *pass = *pass_list, *pass_tmp=NULL;
  
  for( ; pass; pass = pass->next)
    {
      if(strcmp(pass->name, pass_name) == 0)
	return pass;
      else if(pass->sub) 
	{
	  pass_tmp = retreive_pass(pass_name, &pass->sub);
	  if(pass_tmp != NULL)
	    return pass_tmp;
	}
    }
  return NULL;
}

char * search_pass_list_l(struct opt_pass *pass)
{
  bool success = false;

  if(search_pass_list(current_pass, gcc_pass_lists[0]))
    return "all_lowering_passes";
  if(search_pass_list(current_pass, gcc_pass_lists[1]))
    return "all_small_ipa_passes";
  if(search_pass_list(current_pass, gcc_pass_lists[2]))
    return "all_regular_ipa_passes";
  if(search_pass_list(current_pass, gcc_pass_lists[3]))
    return "all_lto_gen_passes";
  if(search_pass_list(current_pass, gcc_pass_lists[4]))
	return "all_passes";
  // Control should never come here
  printf("pass not found\n");
  return "false";
}

cJSON *load_file(char *file_name)
{
  FILE *fp = fopen(file_name, "r");
  if(fp)
    {
      fseek(fp,0,SEEK_END);
      long len=ftell(fp);
      fseek(fp,0,SEEK_SET);
      char *data=xmalloc(len+1);
      fread(data,1,len,fp);
      fclose(fp);
      return cJSON_Parse(data);
    }
  else
    {
      printf("Input file not found...\n");
      exit(1);
    }
}

bool search_pass_list(struct opt_pass *pass_tmp, struct opt_pass **pass_list)
{
  bool success=false;
  struct opt_pass *pass = *pass_list;

  for( ; pass; pass = pass->next)
    {
      if(strcmp(pass->name, pass_tmp->name) == 0)
	success = true;
      if(pass->sub && search_pass_list(pass_tmp, &pass->sub))
	success = true;
    }
  return success;
}
void add_to_json_ipa(struct opt_pass *pass)
{
  cJSON *json_ipa, *json_array, *json_pass, *json_str;

  json_ipa=cJSON_GetObjectItem(j_out, "#ipa#");
  json_array=cJSON_GetObjectItem(json_ipa, search_pass_list_l(pass));
  
  json_pass = cJSON_CreateObject();
  json_str = cJSON_CreateString(pass->name); 
  cJSON_AddItemToObject(json_pass, "pass", json_str);
  json_str = cJSON_CreateString(pass_type(pass->type));
  cJSON_AddItemToObject(json_pass, "type", json_str);

  cJSON_AddItemToArray(json_array, json_pass);
  
}

bool add_to_json(const char *func_name, char *pass_list_name, struct opt_pass *pass, int flag)
{
  cJSON *json_func, *json_array, *json_pass, *json_str;
  
  if(flag == 1)               // new function
    {
      json_func = cJSON_CreateObject();            //function_name

      json_array = cJSON_CreateArray();
      cJSON_AddItemToObject(json_func, "all_lowering_passes", json_array);
      json_array = cJSON_CreateArray();
      cJSON_AddItemToObject(json_func, "all_small_ipa_passes", json_array);
      json_array = cJSON_CreateArray();
      cJSON_AddItemToObject(json_func, "all_lto_gen_passes", json_array);
      json_array = cJSON_CreateArray();
      cJSON_AddItemToObject(json_func, "all_passes", json_array);
      cJSON_AddItemToObject(j_out, func_name, json_func);
    }

  json_func = cJSON_GetObjectItem(j_out, func_name);
  json_array = cJSON_GetObjectItem(json_func, pass_list_name);
  
  json_pass = cJSON_CreateObject();
  json_str = cJSON_CreateString(pass->name);
  cJSON_AddItemToObject(json_pass, "pass", json_str);
  json_str = cJSON_CreateString(pass_type(pass->type));
  cJSON_AddItemToObject(json_pass, "type", json_str);
 
  cJSON_AddItemToArray(json_array, json_pass);
}

char * pass_type(int pass_type)
{
  if(pass_type == 0)
    return "GIMPLE_PASS";
  else if(pass_type == 1)
    return "RTL_PASS";
  else if(pass_type == 2)
    return "SIMPLE_IPA_PASS";
  else if(pass_type == 3)
    return "IPA_PASS";
}

void loop_rec1(struct loop *loop, int *childs)
{
	(*childs) = loop->num;
	childs++;
	if(loop->inner==NULL)
		return ;

	loop_rec1(loop->inner, childs);
}
  
void loop_rec(struct loop *loop, int *childs)
{
  (*childs)++;
  if((loop->inner)==NULL)
    return;

  loop_rec(loop->inner, childs);
}


/************************************* Registering callbacks *************************************/
void finish_plugin(void *gcc_data, void *user_data)
{
  if (strlen(output_file)>0)
     openme_store_json_file(j_out, output_file);
}

/************************************* Registering callbacks *************************************/

/**************************************************************************/
void record_function_names(void *gcc_data, void *user_data)
{
  const char *dname;
  const char *misc;
  void *tmp;
  location_t *loc;
  long line;

  if(current_function_decl != NULL)
  {
    /* Place to write */
    j_tmp=openme_get_object(j_out, ALC_FUNCS);
    if (j_tmp==NULL)
    {
      j_tmp1 = cJSON_CreateArray();
      cJSON_AddItemToObject(j_out, ALC_FUNCS, j_tmp1);
      j_tmp=openme_get_object(j_out, ALC_FUNCS);
    }

    dname = lang_hooks.decl_printable_name(current_function_decl, 2);
    if (dname!=NULL)
    {
      j_tmp1 = cJSON_CreateObject();

      j_tmp2 = cJSON_CreateString(dname);
      cJSON_AddItemToObject(j_tmp1, ALC_COMP_FUNC_NAME, j_tmp2);

      if (main_input_filename!=NULL)
      {
        j_tmp2 = cJSON_CreateString(main_input_filename);
        cJSON_AddItemToObject(j_tmp1, ALC_COMP_SOURCE, j_tmp2);
      }

      if (cfun) 
      {
        expanded_location floc;
        floc = expand_location (DECL_SOURCE_LOCATION (cfun->decl));

        sprintf(buf, "%d", floc.line);
        j_tmp2 = cJSON_CreateString(buf);
        cJSON_AddItemToObject(j_tmp1, "source_start_line", j_tmp2);

        /* hack to detect C and Fortran end of source line*/
        line=LOCATION_LINE(cfun->function_end_locus);
        if (line==0)
           line=LOCATION_LINE(cfun->function_start_locus);

        sprintf(buf, "%d", line);
        j_tmp2 = cJSON_CreateString(buf);
        cJSON_AddItemToObject(j_tmp1, "source_stop_line", j_tmp2);
      }

      cJSON_AddItemToArray(j_tmp, j_tmp1);
    }
  }
}

void init_record_function_names(const char* plugin_name)
{
   // Check if j_out has already been created
   j_tmp = cJSON_GetObjectItem(j_ini, "output_file");
   if (j_tmp==NULL)
   {
     printf("Alchemist error: output_filename is not defined in ini file\n");
     exit(1);
   }

   strcpy(output_file, j_tmp->valuestring);

   j_out=openme_load_json_file(output_file);
   if (j_out==NULL)
      j_out = cJSON_CreateObject();

   register_callback(plugin_name, PLUGIN_ALL_PASSES_START, &record_function_names, NULL);
   register_callback(plugin_name, PLUGIN_FINISH, &finish_plugin, NULL);
}

/**************************************************************************/
void record_executed_passes(void *gcc_data, void *user_data)
{
  const char *dname;
  cJSON *json_tmp, *json_str;
  int flag1 = 0, i;
  bool success=false;

  printf("Current pass: %s (%s)\n", current_pass->name, pass_type(current_pass->type));
  return;
  
  if(current_function_decl == NULL)
    add_to_json_ipa(current_pass);
  else
    {
      dname = lang_hooks.decl_printable_name (current_function_decl, 2);
      
      if(strcmp(dname, func_name) != 0)
	{
	  strcpy(func_name, dname);
	  
	  for(i=0; i<func_count; i++)
	    {
	      //locate if function already exists
	      if(strcmp(funcs_per_file[i], func_name)==0)
		{
		  flag1=1;
		  break;
		}
	    }
	  if(flag1 == 0)            // function not found in functions array
	    {
	      funcs_per_file=(char **)xrealloc(funcs_per_file, (func_count+1)*sizeof(char *));
	      funcs_per_file[func_count++]=xstrdup(func_name);
	      // This is a new function so we have to setup object and array for the function (only done once for each function
	      add_to_json(dname, search_pass_list_l(current_pass), current_pass, 1);
	      flag1=0;                // reset it for any new function
	    }
	  else
	    add_to_json(dname, search_pass_list_l(current_pass), current_pass, 0);
	}
      else
	add_to_json(dname, search_pass_list_l(current_pass), current_pass, 0);
    }
  return;
}

void init_record_executed_passes(const char* plugin_name)
{
   // Check if j_out has already been created
   j_tmp = cJSON_GetObjectItem(j_ini, "output_file");
   if (j_tmp==NULL)
   {
     printf("Alchemist error: output_filename is not defined in ini file\n");
     exit(1);
   }

   strcpy(output_file, j_tmp->valuestring);

   j_out=openme_load_json_file(output_file);
   if (j_out==NULL)
      j_out = cJSON_CreateObject();

   register_callback(plugin_name, PLUGIN_PASS_EXECUTION, &record_executed_passes, NULL);
   register_callback(plugin_name, PLUGIN_FINISH, &finish_plugin, NULL);
}

/************************************* Plugin initialization *************************************/

int plugin_init(struct plugin_name_args *plugin_info,
                struct plugin_gcc_version *version)
{
  /* Setting local plugin variables */
  cJSON *alc_action=NULL;

  cJSON* json_program=NULL;
  cJSON* json_tmp=NULL;

  /* Init OpenME environment (for malloc, free to work inside other programs) */
  openme_init("");

  /* Setting GCC variables */
  struct register_pass_info pass_info_gimple;
  const char *plugin_name=plugin_info->base_name;

  /* Check if file is set */
  str=getenv(alc_env);
  if (str==NULL)
  {
    printf("Alchemist error: environment variable with input file %s is not set !\n", alc_env);
    exit(1);
  }

  /* Reading Alchemist input file */
/*  printf("Loading Alchemist ini file %s ...\n", str); */
  j_ini=openme_load_json_file(str);
  if (j_ini==NULL)
  {
    printf("Alchemist error: failed to load json file %s !\n", str);
    exit(1);
  }

  /* Process actions */
  alc_action=openme_get_object(j_ini, "action");
  if (alc_action==NULL)
  {
    printf("Alchemist error: failed to load json file %s !\n", str);
    exit(1);
  }

  str=alc_action->valuestring;
/*  printf("Initializing Alchemist action %s ...\n", str); */
  
  if (strcmp(str, "record_function_names")==0)
     init_record_function_names(plugin_name);
  else if (strcmp(str, "record_executed_passes")==0)
     init_record_executed_passes(plugin_name);
/*
  else if (strcmp(str, "substitute_passes")==0)
     init_substitute_passes();
  else if (strcmp(str, "record_functions")==0)
     record_code_structure();
*/

  return 0;
  

  j_tmp = cJSON_GetObjectItem(j_ini, "action");
  if(strcmp(json_tmp->valuestring, "record_passes") == 0)
    {
      cJSON *json_ipa;

      j_out = cJSON_CreateObject();
      // Create objects for ipa passes
      json_ipa=cJSON_CreateObject();
      json_tmp = cJSON_CreateArray();
      cJSON_AddItemToObject(json_ipa, "all_regular_ipa_passes", json_tmp);
      json_tmp = cJSON_CreateArray();
      cJSON_AddItemToObject(json_ipa, "all_small_ipa_passes", json_tmp);
      json_tmp = cJSON_CreateArray();
      cJSON_AddItemToObject(json_ipa, "all_lowering_passes", json_tmp);

      cJSON_AddItemToObject(j_out, "#ipa#", json_ipa);

      register_callback(plugin_name, PLUGIN_PASS_EXECUTION, &save_executed_passes, NULL);
      register_callback(plugin_name, PLUGIN_FINISH, &finish_plugin, NULL);
    }
  else if(strcmp(json_tmp->valuestring, "substitute_passes") == 0)
    {
      json_tmp = cJSON_GetObjectItem(j_ini, "input_file");
      strcpy(input_file, json_tmp->valuestring);
      json_tmp = cJSON_GetObjectItem(j_ini, "function_name");
      strcpy(func_name, json_tmp->valuestring);
      // For now, just substitue passes for all_passes list
      register_callback(plugin_name, PLUGIN_ALL_PASSES_START, &substitute_passes, NULL);
      register_callback(plugin_name, PLUGIN_ALL_PASSES_END, &test1, NULL);
      register_callback(plugin_name, PLUGIN_PASS_EXECUTION, &test, NULL);
    }
  else if(strcmp(json_tmp->valuestring, "record_code_structure") == 0)
    {
      struct register_pass_info pass_info_rtl = {
	.pass = &pass_extract_loops.pass,
	.reference_pass_name = "reginfo",
	.ref_pass_instance_number = 0,
	.pos_op = PASS_POS_INSERT_AFTER
      };
      // Check if j_out has already been created
      json_tmp = cJSON_GetObjectItem(j_ini, "output_file");
      strcpy(output_file, json_tmp->valuestring);
      
      if(!file_exist(output_file))
	{
	  j_out = cJSON_CreateObject();
	  json_program = cJSON_CreateArray();
	  cJSON_AddItemToObject(j_out, "#program#", json_program);
	}
      else
	  j_out = load_file(output_file);
      
      register_callback(plugin_name, PLUGIN_PASS_MANAGER_SETUP, NULL, &pass_info_rtl);
      register_callback(plugin_name, PLUGIN_FINISH, &finish_plugin, NULL);
    }
  else if(strcmp(json_tmp->valuestring, "update_memory_ref") == 0)
    {
      pass_info_gimple.pass = &pass_update_memref.pass;
      pass_info_gimple.reference_pass_name="optimized";
      pass_info_gimple.ref_pass_instance_number = 0;
      pass_info_gimple.pos_op = PASS_POS_INSERT_AFTER;
      register_callback(plugin_name, PLUGIN_PASS_MANAGER_SETUP, NULL, &pass_info_gimple);
    }

  json_tmp = cJSON_GetObjectItem(j_ini, "output_file");
  strcpy(output_file, json_tmp->valuestring);
  return 0;
}
