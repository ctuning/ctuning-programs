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

char xyz[1024]="";

/************************************* Misc functions *************************************/

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
    j_tmp=openme_get_obj(j_out, ALC_FUNCS);
    if (j_tmp==NULL)
    {
      j_tmp1 = cJSON_CreateArray();
      cJSON_AddItemToObject(j_out, ALC_FUNCS, j_tmp1);
      j_tmp=openme_get_obj(j_out, ALC_FUNCS);
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

  strcpy(alc_output_file, j_tmp->valuestring);

  j_out=openme_load_json_file(alc_output_file);
  if (j_out==NULL)
     j_out = cJSON_CreateObject();

  register_callback(plugin_name, PLUGIN_ALL_PASSES_START, &record_function_names, NULL);
  register_callback(plugin_name, PLUGIN_FINISH, &finish_plugin, NULL);
}
