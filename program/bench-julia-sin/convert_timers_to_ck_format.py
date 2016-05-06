#
# Converting raw pencil-benchmark timing to CK universal 
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

print ('Converting fine-grain timers from julia-benchmark to CK format ...')

f=open('tmp-output1.tmp','r')
s=f.read()
f.close()

d={}

dcompute=float(s)

if dcompute>0:
   d['execution_time_kernel_0']=dcompute
   d['execution_time']=dcompute

# Write CK json
f=open('tmp-ck-timer.json','wt')
f.write(json.dumps(d, indent=2, sort_keys=True)+'\n')
f.close()
