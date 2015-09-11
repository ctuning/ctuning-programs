#
# Converting raw slambench timing to CK universal 
# autotuning and machine learning format
#
# Collective Knowledge (CK)
#
# See CK LICENSE.txt for licensing details.
# See CK COPYRIGHT.txt for copyright details.
#
# Developer: Grigori Fursin
#

import json

d={}

print ('  (processing characteristics and calculating new ones ...)')

# Preload tmp-ck-timer.json from OpenME if there
exists=True
try:
  f=open('tmp-ck-timer.json', 'r')
except Exception as e:
  exists=False
  pass

if exists:
   try:
     s=f.read()
     d=json.loads(s)
   except Exception as e:
     exists=False
     pass

   if exists:
      f.close()

      et0=d.get('execution_time_kernel_0',0)

      rts=d.get('run_time_state',{})
      dim_n=int(rts.get('dim_n',0))
      dim_nxn=int(rts.get('dim_nxn',0))
      rep=int(rts.get('kernel_repetitions',0))

      if rep!=0:
         et0n=et0/rep
         d['execution_time_kernel_0n']=et0n

         if dim_n!=0 and dim_nxn!=0:
            d['execution_time_kernel_0n_div_by_dim_n']=et0n/dim_n
            d['execution_time_kernel_0n_div_by_dim_nxn']=et0n/dim_nxn
            d['execution_time_kernel_0n_div_by_dim_nxn1']=et0n/(1.5*dim_nxn)

      # Write CK json
      f=open('tmp-ck-timer.json','wt')
      f.write(json.dumps(d, indent=2, sort_keys=True)+'\n')
      f.close()

      print ('  (processed successfully!)')
