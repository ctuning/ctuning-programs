Shared programs, benchmarks and kernels for autotuning/crowd-tuning
===================================================================

Various public benchmarks and kernels that we used in our research on
universal and multi-objective autotuning/crowd-tuning. It is possible
to reproduce and extend some techniques from our recent papers:

* http://arxiv.org/abs/1506.06256
* http://hal.inria.fr/hal-01054763
* https://hal.inria.fr/inria-00436029
* http://arxiv.org/abs/1407.4075

We envision that community will join us in sharing their programs and data sets
to enable systematic, collaborative and reproducible computer engineering.

Benchmarks are considerably simplified to be run on (Android based) 
mobile phones and tables, and have many related data sets.

See some results from crowdsourcing iterative compilation (autotuning) 
on Android-based mobile phones and other computer systems:

* Android App to crowdsource iterative compilation: https://play.google.com/store/apps/details?id=com.collective_mind.node
* Participating mobile phones and tablets: http://cTuning.org/crowdtuning-mobiles
* Processors from above mobile phones: http://cTuning.org/crowdtuning-processors
* Some results from crowdtuning: http://cTuning.org/crowdtuning-results

Status
======
Stable reprository

Dependencies on other repositories
==================================
* ck-autotuning
* ck-env

Authors
=======

* Grigori Fursin, cTuning foundation (France) / dividiti (UK)
* Various authors of shared programs (see individual entries)

Prerequisites
=============
* Collective Knowledge Framework: http://github.com/ctuning/ck

Installation
============

> ck pull repo:ctuning-programs

Get data sets

> ck pull repo:ctuning-datasets-min

Basic Usage
===========

> ck list program

> ck list dataset

> ck compile program:cbench-automotive-susan --speed

> ck run program:cbench-automotive-susan

Add extra data sets per program (at least 20):

Download ckr-ctuning-datasets.zip from https://drive.google.com/folderview?id=0B-wXENVfIO82dzYwaUNIVElxaGc&usp=sharing 
(or other and much larger datasets ckr-usb-ctuning-dataset-* from our PLDI paper).

Register it with CK simply via:

> ck add repo:ctuning-datasets --zip=ckr-ctuning-datasets.zip --quiet

Now, when you run a given program as above, you will have an extended choice of data sets.

If you want to compile and run our benchmarks on Android-based mobile phones,
you need to download and register with CK Android NDK as described here:
* https://github.com/ctuning/ck/wiki/Getting_started_guide_tools
