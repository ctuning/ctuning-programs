import ck.kernel as ck

r=ck.access('list #* out=none')
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

    if 'lang-c' in dd.get('tags',[]):
       dd['extra_ld_vars']='$<<CK_EXTRA_LIB_M>>$'


#    if 'compile_individual_files_cmd' in dd: del(dd['compile_individual_files_cmd'])
#    if 'link_individual_files_cmd' in dd: del(dd['link_individual_files_cmd'])

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
