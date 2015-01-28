/*

 OpenME - Event-driven, plugin-based interactive interface to "open up" 
          any software (C/C++/Fortran/Java/PHP) and possibly connect it to cM

 cM - Collective Mind infrastructure to discover, collect,
      share and reuse knowledge

 Developer(s): (C) Grigori Fursin
 http://cTuning.org/lab/people/gfursin

 */

#ifndef CM_H
#define CM_H

#include <stdio.h>
#include <stdlib.h>
#include <cJSON.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define OPENME_USE "OPENME_USE"
#define OPENME_PLUGINS "OPENME_PLUGINS"
#define OPENME_PLUGIN_INIT_FUNC "openme_plugin_init"
#define OPENME_OUTPUT_FILE "OPENME_OUTPUT_FILE"
#define OPENME_DEBUG "OPENME_DEBUG"

/*
   OpenME event structure

   name - name of event (simple string)
   func - address of a function with 1 void pointer as a parameter
   next - pointer to next event (chaining)
*/

struct openme_event
{
  char *name;
  void (*func) (void *params);
  struct openme_event *next;
};

/*
   OpenME hooks

   We need to update all functions that use malloc and free
   and that we want to use in OpenME library
*/

struct openme_hooks
{
  void *(*malloc)(size_t sz);
  void (*free)(void *ptr);
  FILE *(*fopen)(const char *ptr1, const char *ptr2);
  int (*fprintf)(FILE*, const char*, ...);
  int (*fseek)(FILE *, long, int);
  long (*ftell)(FILE *);
  size_t (*fread)(void *, size_t, size_t, FILE *);
  int (*fclose)(FILE *f1);
};

/*
   OpenME initalization info

   event_list    - point to 1st event (openm_event)
   openme_hooks  - local hooks
   error         - last error message
*/

struct openme_info
{
  struct openme_event *event_list;
  struct openme_hooks *hooks;
  char error[1024];
};

/* OpenME interface functions */
extern int openme_init (char *env_use, char *env_plugins, char *plugin_names, int force_use);

extern void openme_callback (char *event_name, void *params);
extern void openme_register_callback (struct openme_info *ome_info, char *event_name, void *func);

extern cJSON *openme_create_obj (char *str);
extern cJSON *openme_get_obj (cJSON*json, char *str);
extern void openme_print_obj (cJSON *obj);

extern cJSON *cm_action(cJSON *inp);

extern void openme_init_hooks(struct openme_hooks *ome_hooks);

extern cJSON *openme_load_json_file(char *file_name);
extern int openme_store_json_file(cJSON *json, char *file_name);
extern void openme_set_error(char *format, char *text);
extern void openme_print_error(void);

/* Fortran interface */
extern int openme_init_f_ (char *env_use, char *env_plugins, char *plugin_names, int force_use);
extern int OPENME_INIT_F (char *env_use, char *env_plugins, char *plugin_names, int force_use);

extern void openme_callback_f (char *event_name, void *params);
extern void OPENME_CALLBACK_F (char *event_name, void *params);

extern cJSON *openme_create_obj_f_ (char *str);
extern cJSON *OPENME_CREATE_OBJ_F (char *str);

extern cJSON *openme_get_obj_f_ (char *str);
extern cJSON *OPENME_GET_OBJ_F (char *str);

extern void openme_print_obj_f_ (cJSON **obj);
extern void OPENME_PRINT_OBJ_F (cJSON **obj);

#ifdef __cplusplus
}
#endif

#endif
