/*

 Alchemist headers

 Developer(s): (C) Grigori Fursin, started on 2011.09

 */

#ifndef ALCHEMIST_SUBSTITUTE_PASSES_H
#define ALCHEMIST_SUBSTITUTE_PASSES_H

char *pass_group(struct opt_pass *pass);
char *pass_type(int pass_type);
void view_pass_execution(void *gcc_data, void *user_data);
struct opt_pass *find_pass(char *pass_name, struct opt_pass **pass_list);
bool search_pass_list(struct opt_pass *pass_tmp, struct opt_pass **pass_list);

void record_executed_passes(void *gcc_data, void *user_data);
void init_record_executed_passes(const char* plugin_name);

void substitute_passes_end(void *gcc_data, void *user_data);
void substitute_passes(void *gcc_data, void *user_data);
void init_substitute_passes(const char* plugin_name);

#endif
