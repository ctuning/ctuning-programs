/*

 Alchemist high-level plugin to open up GCC
 for interactive (online) application analysis and tuning

 License: GNU GPL v3

 Developer(s): (C) Grigori Fursin, started on 2011.09
 http://cTuning.org/lab/people/gfursin

*/

#include "gcc-plugin.h"

#include <openme.h>
#include <cJSON.h>

#include "alchemist.h"

int plugin_is_GPL_compatible;

struct gcc_init_data 
{
  struct plugin_name_args *plugin_info;
  struct plugin_gcc_version *version;  
};

/************************************* Plugin initialization *************************************/

int plugin_init(struct plugin_name_args *plugin_info,
                struct plugin_gcc_version *version)
{
  static cJSON *dummy=NULL;

  struct gcc_init_data tmp={plugin_info,version};

  /* needed to link properly cjson lib */
  dummy = cJSON_CreateArray();

  /* Init OpenME environment (for malloc, free to work inside other programs) */
  openme_init(NULL, ALC_PLUGINS, NULL, 1);

  openme_callback("SUB_PLUGIN_INIT", &tmp);

  return 0;
}
