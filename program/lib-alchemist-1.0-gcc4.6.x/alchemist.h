/*

 Alchemist headers

 Developer(s): (C) Grigori Fursin, started on 2011.09

 */

#ifndef ALCHEMIST_H
#define ALCHEMIST_H

#define ALC_FUNCS "alchemist_functions"
#define ALC_PASSES "alchemist_passes"
#define ALC_PASSES_IPA "#ipa#"

#define ALC_COMP_FUNC_NAME          "name"
#define ALC_COMP_SOURCE             "source_filename"
#define ALC_COMP_FUNC_START_LINE    "source_start_line"
#define ALC_COMP_FUNC_STOP_LINE     "source_stop_line"

void finish_plugin(void *gcc_data, void *user_data);

#endif
