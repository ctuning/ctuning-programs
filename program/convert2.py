import ck.kernel as ck

r=ck.access('list #milepost* out=none')
if r['return']>0: 
   print 'Error: '+r['error']
   exit(1)

lst=r['lst']

r=ck.load_json_file({'json_file':'convert2.json'})
if r['return']>0:
   print 'Error: '+r['error']
   exit(1)
d1=r['dict']

dtags={}

for q in lst:
    m=q['module_uoa']
    d=q['data_uoa']

    print ('***********************************')
    print (d)

    r=ck.access({'action':'load',
                 'module_uoa':m,
                 'data_uoa':d})
    if r['return']>0: 
       print 'Error: '+r['error']
       exit(1)

    dd=r['dict']
    dm=r['info']
    dn=r['data_name']

    dd.update(d1)

    if 'add_target_extension' in dd: del(dd['add_target_extension'])
    if 'build_scripts_names' in dd: del(dd['build_scripts_names'])
    if 'build_scripts_uoa' in dd: del(dd['build_scripts_uoa'])
    if 'clean_scripts_names' in dd: del(dd['clean_scripts_names'])
    if 'cm_build_dependencies' in dd: del(dd['cm_build_dependencies'])
    if 'cm_classes_uoa' in dd: del(dd['cm_classes_uoa'])

    tags=dd.get('tags',[])

    ml=dd.get('main_language','').split(',')
    if 'c' in ml: tags.append('lang-c')
    if 'fortran' in ml: tags.append('lang-f77')

#    tags.append('small')
#    tags.append('crowd-tuning')

    dd['tags']=list(set(tags))

    cd=dd.get('compile_deps',{})

    if 'codelet' in tags:
       if 'rtl_milepost_codelet' not in cd:
           cd['rtl_milepost_codelet']={"local": "yes", "tags":"lib,rtl,milepost,codelet"}
    else:
       if 'rtl_milepost_codelet' in cd:
          del(cd['rtl_milepost_codelet'])

    if 'polybench' in tags:
       if 'rtl_polybench' not in cd:
           cd['rtl_polybench']={"local": "yes", "tags":"lib,rtl,polybench"}

    dd['compile_deps']=cd

    run_cmds=dd.get('run_cmds',{})
    for rc in run_cmds:
        x=run_cmds[rc]
        if x.get('ignore_return_code','')=='': x['ignore_return_code']='no'

        y=x.get('run_time',{})
        z=y.get('run_cmd_main','')

        if not z.startswith('$#BIN_FILE#$'): z='$#BIN_FILE#$ '+z

        ix=z.find(' codelet.data')
        if ix>0:
           z=z.replace(' codelet.data', ' $#src_path#$codelet.data')

        ix=z.find(' $#src_path#$codelet.data')
        if ix>0:
           z=z.replace(' $#src_path#$codelet.data', ' \"$#src_path#$codelet.data\"')
           

        y['run_cmd_main']=z

        dt=x.get('dataset_tags',[])
        dcu=x.get('dataset_classes_uoa',[])

        if len(dcu)>0:
           for k in dcu:
               if k not in dtags:
                  ck.out('')
                  ra=ck.inp({'text':'Enter dataset tag for '+k+': '})
                  kk=ra['string'].strip().lower()
                  dtags[k]=kk
               kk=dtags[k]
               if kk not in dt: dt.append(kk)

           del(x['dataset_classes_uoa'])

        dt.append('dataset')
        x['dataset_tags']=list(set(dt))

        if 'codelet' in tags:
           del(x['dataset_tags'])

    r=ck.access({'action':'update',
                 'module_uoa':m,
                 'data_uoa':d,
                 'dict':dd,
                 'sort_keys':'yes',
                 'ignore_update':'yes',
                 'substitute':'yes'})
    if r['return']>0: 
       print 'Error: '+r['error']
       exit(1)
