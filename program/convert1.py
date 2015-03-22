import ck.kernel as ck

r=ck.access('list # out=none')
if r['return']>0: 
   print 'Error: '+r['error']
   exit(1)

for q in r['lst']:
    m=q['module_uoa']
    d=q['data_uoa']

    r=ck.access({'action':'load',
                 'module_uoa':m,
                 'data_uoa':d})
    if r['return']>0: 
       print 'Error: '+r['error']
       exit(1)

    dd=r['dict']
    dm=r['info']
    dn=r['data_name']

    dn1=''
    if dn.startswith('milepost'):
       dn1=dn.replace(':', '-')
       dn1=dn1.replace(' ', '-')
       dn1=dn1.replace('--', '-')
    elif dn.startswith('benchmark-'):
       dn1=dn[10:]

    if dn1!='':
       rx=ck.access({'action':'rename',
                     'module_uoa':m,
                     'data_uoa':d,
                     'new_data_uoa':dn1})
       if rx['return']>0:
          print 'Error: '+rx['error']
          exit(1)

    
