/*

 OpenME - Event-driven, plugin-based interactive interface to "open up" 
          any software (C/C++/Fortran/Java/PHP) and possibly connect it to cM

 cM - Collective Mind infrastructure to discover, collect,
      share and reuse knowledge

 Developer(s): (C) Grigori Fursin
 http://cTuning.org/lab/people/gfursin

 This library is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation; either
 version 2.1 of the License, or (at your option) any later version.

 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public
 License along with this library; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "openme.h"

#ifdef WINDOWS
#include <windows.h>
static HINSTANCE dll=NULL; 
#else
#include <dlfcn.h>
static void *dll;
#endif

/* set 'use' to 1 to process callbacks */
static int openme_use=0;
static int openme_initialized=0;

static struct openme_hooks oh={NULL,NULL,NULL,NULL};
static struct openme_info oi={NULL,NULL,""};

static char bufx[1024];
static char *bufy;
static cJSON *json;
static cJSON *json1;

extern int openme_init (char *env_use, char *env_plugins, char *plugin_names, int force_use)
{
  /*
     Initialize OpenME plugins

     Input: env_use -      if not NULL and not "", use this environment variable to check if openme is used or not
                           (i.e. if environment variable is set to 1, openme will be used).
                           This can be useful to transparently turn on or off usage of openme in tools or programs
                           (for example, transparent collective optimization)
            env_plugins  - if not NULL and not "", use this environment variable to describe/load plugins
                           (separated by ;)
            plugin_names - if not NULL and not "", use these plugins separated by ; (instead of checking in environment)
            force_use    - if 1, force usage of openme (skip checks for env_use and env_plugins)
                           if -1, just initialize malloc/free
                           0 - standard usage

     Output: 0 - if success
  */

  char *env=NULL;
  char *p;
  char *buf, *buf1, *buf2;
  int i=0;

  char xenv[1028];

  void (*func) (void*);

  if ( ((env = getenv(OPENME_DEBUG)) != NULL) && (atoi(env)==1) )
  {
    fflush(stdout); 
    printf ("Initializing OpenME\n");
  }

  /* Init malloc and free hooks to initialize memory in user space */
  if (oh.malloc==NULL)
  {
     oh.malloc=malloc;
     oh.free=free;
     oh.fopen=fopen;
     oh.fprintf=fprintf;
     oh.fseek=fseek;
     oh.ftell=ftell;
     oh.fread=fread;
     oh.fclose=fclose;
     oi.hooks=&oh;
  }

// Various events from various plugins can coexist
//  oi.event_list=NULL;

  openme_initialized=1;

  /* Check which env variable to use */
  strcpy(xenv, OPENME_USE);
  if ((env_use!=NULL) && (strlen(env_use)>0)) strcpy(xenv, env_use);

  if (force_use!=-1)
  {
    if ((force_use==1) || ((env = getenv(xenv)) != NULL))
    {
      if ((force_use==1) || (atoi(env)==1))
      {
        /* Check which env variable to use */
        strcpy(xenv, OPENME_PLUGINS);
        if ((env_plugins!=NULL) && (strlen(env_plugins)>0)) strcpy(xenv, env_plugins);

        if ((plugin_names!=NULL) && (strlen(plugin_names)>0))
          env=plugin_names;
        else if ((env=getenv(xenv))==NULL)
          return 1;

        buf=malloc(sizeof(char)*(strlen(env)+1));
        buf1=env;

        while (*buf1!=0)
        {
          buf2=buf;
          while ((*buf1!=0) && (*buf1!=';'))
            *buf2++=*buf1++;
          *buf2=0;
          if (*buf1!=0) buf1++;

  #ifdef WINDOWS
          dll = LoadLibrary(TEXT(buf));
          if (dll == NULL)
              printf("Error: Failed to load plugin (error=%u)!\n", GetLastError());
  #else
          dll=dlopen(env, RTLD_LAZY);
          if (dll == NULL)
              printf("Error: Failed to load plugin (%s)!\n", dlerror());
  #endif
          if (dll == NULL)
          {
              free(buf);
              return 1;
          }

  #ifdef WINDOWS
          func = (void (__cdecl *)(void *)) GetProcAddress(dll, OPENME_PLUGIN_INIT_FUNC);
  #else
          func = dlsym (dll, OPENME_PLUGIN_INIT_FUNC);
  #endif
          if (func == NULL)
          {
              printf("Error: Can't find openme_plugin_init function in the plugin!\n");
              free(buf);
              return 1;
          }

          (*func) (&oi);

          openme_use=1;
        }

  //      free(buf);
      }
    }
  }

  return 0;
}

extern void openme_init_hooks(struct openme_hooks *hooks)
{
 /* Set local malloc and free to user program's malloc and free
    to be able to allocate memory in plugins */

  oh.malloc=hooks->malloc;
  oh.free=hooks->free;
  oh.fopen=hooks->fopen;
  oh.fprintf=hooks->fprintf;
  oh.fseek=hooks->fseek;
  oh.ftell=hooks->ftell;
  oh.fread=hooks->fread;
  oh.fclose=hooks->fclose;
  oi.hooks=&oh;
}

extern void openme_callback (char *event_name, void *params)
{
  /* 
     Initiate callback

     Input: event_name   - name of the event. For now it's just a string and the search is not optimized.
                           We should add cM UIDs here too since events can be possibly shared across many plugins.
            params       - parameters passed to an event. If there are multiple parameters, 
                           we use either struct or cJSON similar to cM universal call

     Output: None - if we need to return some info from the event, we update variable "params".
                    For example, to update unroll factor in LLVM, Open64 ot GCC, we use a struct
                    with unroll factor
  */

  char *env=NULL;
  struct openme_event *es;

  if (openme_initialized==0)
  {
    fflush(stdout); 
    printf ("openme error: callback is used before init!\n");
    fflush(stdout); 
    exit(1);
  }

  if (openme_use==1)
  {
    if ( ((env = getenv(OPENME_DEBUG)) != NULL) && (atoi(env)==1) )
    {
      fflush(stdout);
      printf ("Searching event=%s\n", event_name);
    }
    es=oi.event_list;
    while (es!=NULL)
    {
      if (strcmp(es->name, event_name)==0)
      {
        (*(es->func)) (params);
        /* Don't stop here to allow the same events 
           called from other plugins - may be useful */
      }
      es=es->next;
    }
  }
}

extern void openme_register_callback (struct openme_info *info, char *event_name, void *func)
{
  /*
     Register callback

     Input: info         - plugin initalization variable - passed to plugin init function
            event_name   - name of the event. For now it's just a string and the search is not optimized.
                           We should add cM UIDs here too since events can be possibly shared across many plugins.
            func         - address of the event function.

     Output: None
  */

  char *env=NULL;
  struct openme_event *e, *es;
  char *en;

//  FGG removed it since we now have customized initializer ...
//  if (openme_use==0 && openme_initialized==0)
//     openme_init("");
  if ( ((env = getenv(OPENME_DEBUG)) != NULL) && (atoi(env)==1) )
  {
    fflush(stdout); 
    printf("Register callback %s\n", event_name);
  }

  if ((info==NULL) || (info->hooks==NULL))
  {
     printf("Error: attempt to use OpenME plugin before init\n");
     exit(1);
  }

  en=(char *) info->hooks->malloc(sizeof(char)*(strlen(event_name)+1));
  strcpy(en, event_name);

  e=(struct openme_event *) info->hooks->malloc(sizeof(struct openme_event));
  e->name=en;
  e->func=func;
  e->next=NULL;

  if (info->event_list==NULL)
    info->event_list=e;
  else
  {
    es=info->event_list;
    while ((es->next)!=NULL)
      es=es->next;

    es->next=e;
  }
}

/* Some functions to unify access to json structures so that we can use it in any language
   such as C, C++, Fortran, Java, etc; not completed */

extern cJSON *openme_create_obj (char *str)
{
  /*
     Create object {"a":"b"}

     Input: str   - string of format "a=b c=d ... (@file.json)"
                    if @ is in the string, load and parse json file after @

     Output: Pointer to created object
  */

  char str1[1024];
  char str2[1024];
  int i=0;
  int j=0;
  int il;
  int k, kl;

  cJSON *obj, *obj1, *obj2;

  obj=cJSON_CreateObject();

  il=strlen(str);
  while (i<il)
  {
    if (str[i]=='@')
    {
      /* Process file */
      i++;

      while (str[i]!=' ' && i<il)
        str1[j++]=str[i++];
      str1[j]=0;

      obj1=openme_load_json_file(str1);
      if (obj1==NULL)
         return NULL;

      kl=cJSON_GetArraySize(obj1);
      for (k=0; k<kl; k++)
      {
        if (cJSON_GetObjectItem(obj, cJSON_GetArrayItemName(obj1,k))!=NULL)
          cJSON_DeleteItemFromObject(obj, cJSON_GetArrayItemName(obj1,k));
        cJSON_AddItemReferenceToObject(obj, cJSON_GetArrayItemName(obj1,k), cJSON_GetArrayItem(obj1,k));
      }
    }
    else
    {
      j=0;
      while (str[i]!='=' && i<il)
        str1[j++]=str[i++];
      str1[j]=0;

      i++;
      if (i>=il) break;

      j=0;
      while (str[i]!=' ' && i<il)
        str2[j++]=str[i++];
      str2[j]=0;
      i++;

      cJSON_AddStringToObject(obj, str1, str2);
    }
  }

  return obj;
}

extern cJSON *openme_get_obj(cJSON *json, char *str)
{
  /*
     Get pointer to object by name

     Input: json - current sub-object
            str  - parameter name

     Output: Pointer to found object or NULL
  */

   return cJSON_GetObjectItem(json, str);
}

extern void openme_print_obj (cJSON *obj)
{
  /*
     Print cJSON object

     Input: json - object to print

     Output: None
  */

  printf("%s\n", cJSON_Print(obj));
}

extern cJSON *openme_load_json_file(char *file_name)
{
  /*
     Load json file and create cJSON object

     Input: file_name - name of the file to load

     Output: cJSON object or NULL
  */

  cJSON *obj;
  FILE *fp;
  long len;
  char *data;

  fp = oh.fopen(file_name, "r");
  if(!fp)
  {
    openme_set_error("OpenME error: can't find json file %s\n", file_name);
    return NULL;
  }

  oh.fseek(fp,0,SEEK_END);
  len=oh.ftell(fp);
  oh.fseek(fp,0,SEEK_SET);
  data=oh.malloc(len+1);
  oh.fread(data,1,len,fp);
  oh.fclose(fp);

  obj=cJSON_Parse(data);
  oh.free(data);

  return obj;
}

extern int openme_store_json_file(cJSON *json, char *file_name)
{
  /*
     Store json object in file

     Input:  json      - cJSON object
             file_name - name of the file to store json object

     Output: int r = 0 - if correct; >0 if error
  */

  FILE *fp = oh.fopen(file_name, "w");
  if(!fp)
  {
    openme_set_error("OpenME error: can't open json file %s for writing ...\n", file_name);
    return 1;
  }
  printf("%s\n", cJSON_Print(json));
  if (oh.fprintf(fp, "%s\n", cJSON_Print(json))<0)
  {
    openme_set_error("OpenME error: problems writing to json file %s ...\n", file_name);
    return 1;
  }
  oh.fclose(fp);

  return 0;
}

extern void openme_set_error(char *format, char *text)
{
  /*
     Set OpenME error

     Input:  format - printf format
             text   - error text

     Output: None
  */

  sprintf(oi.error, format, text);
}

extern void openme_print_error(void)
{
  /*
     Print OpenME error

     Input:  None

     Output: None
  */

  printf(oi.error);
}

extern cJSON *cm_action (cJSON *inp)
{
  /*
     FGG: TBD - call local cM

     Input:  inp - json object

     Output: json object from cM or NULL if error (openme error is set)
  */
  char *fn;
  int r=0;

  char fn1[128];
  char fn2[128];
  char fn3[128];

  /* Get module name */
  json=openme_get_obj(inp, "cm_run_module_uoa");
  if (json==NULL)
  {
    openme_set_error("OpenME error - can't find cm_run_module_uoa in cm_action ...", NULL);
    return NULL;
  }
  bufy=json->valuestring;

  /* Generate tmp files with json and for output*/
  /* First file will be deleted automatically by cM */
  fn=tmpnam(NULL);
  sprintf(fn1, "%s-cm.tmp", fn);

  fn=tmpnam(NULL);
  sprintf(fn2, "%s-cm.tmp", fn);

  fn=tmpnam(NULL);
  sprintf(fn3, "%s-cm.tmp", fn);

  /* Record input file */
  r=openme_store_json_file(inp, fn1);
  if (r>0)
  {
    openme_set_error("OpenME error - can't save tmp file ...", NULL);
    return NULL;
  }

  /* Prepare command line */
  sprintf(bufx, "cm %s @%s > %s 2> %s", bufy, fn1, fn2, fn3);

  system(bufx);

  /* Try to read stdout */
  json=openme_load_json_file(fn2);
  if (json==NULL)
  {
    /* FGG TODO: We should add reading of stdout and stderr and put it here in case of error */
    sprintf(bufx, "STDOUT file=%s; STDERR file=%s", fn2);
    sprintf(bufx, "STDOUT file=%s; STDERR file=%s", fn3);

    openme_set_error("OpenME error - can't parse cM output; see files: %s...", bufx);

    return NULL;
  }

  /* Remove tmp files */
  remove(fn2);
  remove(fn3);

  return json;
}

/* Fortran interface for OpenME */

extern int openme_init_f_ (char *env_use, char *env_plugins, char *plugin_names, int force_use) 
  {return openme_init(env_use, env_plugins, plugin_names, force_use);}
extern int OPENME_INIT_F (char *env_use, char *env_plugins, char *plugin_names, int force_use) 
  {return openme_init(env_use, env_plugins, plugin_names, force_use);}

extern void openme_callback_f_ (char *event_name, void *params) {openme_callback(event_name, params);}
extern void OPENME_CALLBACK_F (char *event_name, void *params) {openme_callback(event_name, params);}

extern cJSON *openme_create_obj_f_ (char *str) {return openme_create_obj(str);}
extern cJSON *OPENME_CREATE_OBJ_F (char *str) {return openme_create_obj(str);}

extern void openme_print_obj_f_ (cJSON **obj) {openme_print_obj(*obj);}
extern void OPENME_PRINT_OBJ_F (cJSON **obj) {openme_print_obj(*obj);}

extern cJSON *cm_action_f_ (cJSON **obj) {cm_action(*obj);}
extern cJSON *CM_ACTION_F (cJSON **obj) {cm_action(*obj);}
