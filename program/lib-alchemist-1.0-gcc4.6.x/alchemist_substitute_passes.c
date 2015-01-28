/*

 Alchemist low-level plugin for GCC

 Author(s): Abdul Wahid Memon
            Grigori Fursin (http://cTuning.org/lab/people/gfursin)

 Started on 2011.09
*/

#include "gcc-plugin.h"

#include <stdio.h>

//#include "config.h"
//#include "system.h"
//#include "coretypes.h"
//#include "tm.h"
//#include "toplev.h"
#include "gimple.h"
#include "langhooks.h"
//#include "function.h"
//#include "cfgloop.h"
//#include "tree.h"
#include "tree-pass.h"
//#include "tree-flow.h"
//#include "tree-inline.h"
//#include "tree-flow-inline.h"
//#include "version.h"
#include "input.h"

#include <openme.h>
#include <cJSON.h>

#include "alchemist.h"
#include "alchemist_common.h"

/************************************* Misc functions *************************************/
void view_pass_execution(void *gcc_data, void *user_data)
{
  const char *dname;

  if(current_function_decl != NULL)
    {
      dname = lang_hooks.decl_printable_name (current_function_decl, 2);

      if (dname!=NULL)
	printf("Really executing %s - %s ...\n", dname, current_pass->name);
    }
}

bool search_pass_list(struct opt_pass *pass_tmp, struct opt_pass **pass_list)
{
  bool success=false;
  struct opt_pass *pass = *pass_list;

  for(; pass; pass = pass->next)
  {
    if(strcmp(pass->name, pass_tmp->name) == 0)
    {
      success = true;
      break;
    }

    if(pass->sub && search_pass_list(pass_tmp, &pass->sub))
    {
      success = true;
      break;
    }
  }
  return success;
}

char *pass_group(struct opt_pass *pass)
{
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

  /* Control should never come here */
  printf("pass not found\n");
  return "false";
}

char *pass_type(int pass_type)
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

struct opt_pass *find_pass(char *pass_name, struct opt_pass **pass_list)
{
  struct opt_pass *pass = *pass_list, *pass_tmp=NULL;
  
  for( ; pass; pass = pass->next)
    {
      if(strcmp(pass->name, pass_name) == 0)
	return pass;
      else if(pass->sub) 
	{
	  pass_tmp = find_pass(pass_name, &pass->sub);
	  if(pass_tmp != NULL)
	    return pass_tmp;
	}
    }

  return NULL;
}

/************************************* Registering callbacks *************************************/

/**************************************************************************/
void record_executed_passes(void *gcc_data, void *user_data)
{
  const char *dname;

  /* Place to write */
  j_tmp=(cJSON*) openme_get_obj(j_out, ALC_PASSES);
  if (j_tmp==NULL)
  {
    j_tmp3 = cJSON_CreateObject();
    cJSON_AddItemToObject(j_out, ALC_PASSES, j_tmp3);
    j_tmp=openme_get_obj(j_out, ALC_PASSES);
  }

  /* Check if IPA */
  strcpy(buf, ALC_PASSES_IPA);
  if(current_function_decl != NULL)
  {
    dname = lang_hooks.decl_printable_name(current_function_decl, 2);
    if (dname!=NULL)
    {
       strcpy(buf, dname);
    }
  }

  /* Function */
  j_tmp1=openme_get_obj(j_tmp, buf);
  if (j_tmp1==NULL)
  {
    j_tmp3 = cJSON_CreateObject();
    cJSON_AddItemToObject(j_tmp, buf, j_tmp3);
    j_tmp1=openme_get_obj(j_tmp, buf);
  }

  /* Pass group */
  strcpy(buf, pass_group(current_pass));
  j_tmp2=openme_get_obj(j_tmp1, buf);
  if (j_tmp2==NULL)
  {
    j_tmp3 = cJSON_CreateArray();
    cJSON_AddItemToObject(j_tmp1, buf, j_tmp3);
    j_tmp2=openme_get_obj(j_tmp1, buf);
  }

  j_tmp3 = cJSON_CreateObject();
  j_tmp4 = cJSON_CreateString(current_pass->name); 
  cJSON_AddItemToObject(j_tmp3, "pass", j_tmp4);
  j_tmp4 = cJSON_CreateString(pass_type(current_pass->type));
  cJSON_AddItemToObject(j_tmp3, "type", j_tmp4);

  cJSON_AddItemToArray(j_tmp2, j_tmp3);
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

  strcpy(alc_output_file, j_tmp->valuestring);

  j_out=openme_load_json_file(alc_output_file);
  if (j_out==NULL)
     j_out = cJSON_CreateObject();

  register_callback(plugin_name, PLUGIN_PASS_EXECUTION, &record_executed_passes, NULL);
  register_callback(plugin_name, PLUGIN_FINISH, &finish_plugin, NULL);
}

/**************************************************************************/
void substitute_passes_end(void *gcc_data, void *user_data)
{
  printf("Restoring all_passes ...\n");
  if (save_all_passes!=NULL) all_passes=save_all_passes;
}

void substitute_passes(void *gcc_data, void *user_data)
{
  const char *dname;
  int number_of_passes;
  struct opt_pass *pass_tmp;

  if (save_all_passes==NULL) save_all_passes=all_passes;

  /* For now we deal only with ALL PASSES per function */
  if(current_function_decl != NULL)
  {
    dname = lang_hooks.decl_printable_name(current_function_decl, 2);
    if (dname!=NULL)
    {
      /* Function */
      j_tmp1=openme_get_obj(j_in, dname);
      if (j_tmp1!=NULL)
      {
        /* Passes */
        j_tmp2=openme_get_obj(j_tmp1, "all_passes");
        if (j_tmp2==NULL)
        {
          printf("Alchemist error: can't find all_passes for function '%s' in input file ...\n", dname);
          exit(1);
        }

        printf("\nSubstituting passes for function %s ...\n\n", dname);

        number_of_passes = cJSON_GetArraySize(j_tmp2);
        j_tmp3 = j_tmp2->child;

        while(number_of_passes > 0)
        {
          j_tmp4 = cJSON_GetObjectItem(j_tmp3, "pass");
          if (j_tmp4==NULL)
          {
            printf("Alchemist error: pass structure is broken in input file for function %s...\n", dname);
            exit(1);
          }

          strcpy(buf, j_tmp4->valuestring);
          if (strcmp(buf, "*clean_state")!=0)
          {
             pass_tmp = find_pass(buf, &all_passes);
             if(pass_tmp)
             {
                printf("Executing pass %s, %s\n", buf, pass_tmp->name);
                fflush(stdout);
                execute_one_pass(pass_tmp);
             }
             else
             {
               printf("Alchemist error: pass %s was not found in GCC...\n", pass_tmp->name);
               exit(1);
             }
          }

          j_tmp3 = j_tmp3->next;
          number_of_passes--;
        }

        printf("\nFinishing substitution ...\n");

        /* Pointing real pass list to the final pass */
        all_passes = find_pass("*clean_state", &all_passes);
      }
    }
  }
}

void init_substitute_passes(const char* plugin_name)
{
  // Check if j_out has already been created
  j_tmp = cJSON_GetObjectItem(j_ini, "input_file");
  if (j_tmp==NULL)
  {
    printf("Alchemist error: input_filename is not defined in ini file\n");
    exit(1);
  }

  strcpy(alc_input_file, j_tmp->valuestring);

  j_in=openme_load_json_file(alc_input_file);
  if (j_in==NULL)
  {
    printf("Alchemist error: problem loading input file %s ...\n", alc_input_file);
    exit(1);
  }

  j_in=openme_get_obj(j_in, ALC_PASSES);
  if (j_in==NULL)
  {
    printf("Alchemist error: can't find id '%s' in input file ...\n", ALC_PASSES);
    exit(1);
  }

  register_callback(plugin_name, PLUGIN_ALL_PASSES_START, &substitute_passes, NULL);
  register_callback(plugin_name, PLUGIN_ALL_PASSES_END, &substitute_passes_end, NULL);
  register_callback(plugin_name, PLUGIN_PASS_EXECUTION, &view_pass_execution, NULL);
}
