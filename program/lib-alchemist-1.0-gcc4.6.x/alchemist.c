/*

 Alchemist low-level plugin to open up GCC
 for interactive (online) application analysis and tuning

 License: GNU GPL v3

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
#include "alchemist_substitute_passes.h"
#include "alchemist_record_function_names.h"

struct gcc_init_data 
{
  struct plugin_name_args *plugin_info;
  struct plugin_gcc_version *version;  
};

/* Alchemist input file environment */
char *alc_env="CM_ALCHEMIST_INPUT_FILE";

/* Alchemist ini file */
cJSON *j_ini=NULL;

/* Alchemist aggregated output */
char alc_output_file[1024];
cJSON *j_out=NULL;
char alc_input_file[1024];
cJSON *j_in=NULL;

/* Alchemist various buffers and temp vars*/
char *str=NULL;
char buf[1024];

cJSON *j_tmp=NULL; 
cJSON *j_tmp1=NULL; 
cJSON *j_tmp2=NULL; 
cJSON *j_tmp3=NULL; 
cJSON *j_tmp4=NULL; 

struct opt_pass *save_all_passes=NULL;

int plugin_is_GPL_compatible;

/************************************* Registering callbacks *************************************/
void finish_plugin(void *gcc_data, void *user_data)
{
  if (strlen(alc_output_file)>0)
     openme_store_json_file(j_out, alc_output_file);
}

/************************************* Plugin initialization *************************************/

int sub_plugin_init1(struct plugin_name_args *plugin_info,
                struct plugin_gcc_version *version)
{
  /* Setting local plugin variables */
  cJSON *alc_action=NULL;

  cJSON* json_program=NULL;
  cJSON* json_tmp=NULL;

  /* If we want to use openme functions here, we need to initialize OpenMe
     - however it seems to be needed only for Windows */
#ifdef WINDOWS
  openme_init (NULL,NULL,NULL,-1);
#endif

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
  alc_action=openme_get_obj(j_ini, "action");
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
  else if (strcmp(str, "substitute_passes")==0)
     init_substitute_passes(plugin_name);
  else
  {
    printf("Alchemist error: action \"%s\" is not recognized!\n", str);
    exit(1);
  }
/*
  else if (strcmp(str, "record_code_structure")==0)
     init_record_code_structure(plugin_name);
  else if (strcmp(str, "break_code_semantics_to_observe_reaction")==0)
     init_break_code_to_observe_reaction(plugin_name); */

  return 0;
}

int sub_plugin_init(struct gcc_init_data *gid)
{
  struct plugin_name_args *plugin_info=gid->plugin_info;
  struct plugin_gcc_version *version=gid->version;  

  const char *plugin_name=gid->plugin_info->base_name;

  sub_plugin_init1(plugin_info, version);

  return 0;
}

extern
#ifdef WINDOWS
__declspec(dllexport) 
#endif
int openme_plugin_init(struct openme_info *ome_info)
{
  cJSON *dummy=NULL; 

  openme_register_callback(ome_info, "SUB_PLUGIN_INIT", sub_plugin_init);

  dummy=cJSON_CreateObject();

  return 0;
}

