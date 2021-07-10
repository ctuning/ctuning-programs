Shared programs, benchmarks and kernels for autotuning/crowd-tuning
===================================================================

**All CK components can be found at [cKnowledge.io](https://cKnowledge.io) and in [one GitHub repository](https://github.com/ctuning/ck-mlops)!**

[![compatibility](https://github.com/ctuning/ck-guide-images/blob/master/ck-compatible.svg)](https://github.com/ctuning/ck)

These are various public programs, benchmarks and kernels used
in our research on universal and multi-objective autotuning/crowd-tuning
in the [open Collective Knowledge format](http://cKnowledge.org):

* http://arxiv.org/abs/1506.06256
* http://hal.inria.fr/hal-01054763
* https://hal.inria.fr/inria-00436029
* http://arxiv.org/abs/1407.4075

We envision that community will join us in sharing their programs and data sets
to enable systematic, collaborative and reproducible computer engineering.

Benchmarks are considerably simplified to be run on Linux, Windows, MacOs 
and even on Android based mobile phones and tables together with 
[open CK data sets](https://github.com/ctuning/ctuning-datasets-min).

See some results from crowdsourcing iterative compilation (autotuning) 
on Android-based mobile phones and other computer systems:

* Android App to crowdsource iterative compilation: http://cKnowledge.org/android-apps.html
* Participating mobile phones and tablets: http://cTuning.org/crowdtuning-mobiles
* Processors from above mobile phones: http://cTuning.org/crowdtuning-processors
* Some results from crowdtuning: http://cTuning.org/crowdtuning-results

Status
======
Stable reprository

Dependencies on other repositories
==================================
* [ck-autotuning](https://github.com/ctuning/ck-autotuning)
* [ck-env](https://github.com/ctuning/ck-env)

Authors
=======

* [Grigori Fursin](https://fursin.net), cTuning foundation
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

Publications
============

```
@inproceedings{ck-date16,
    title = {{Collective Knowledge}: towards {R\&D} sustainability},
    author = {Fursin, Grigori and Lokhmotov, Anton and Plowman, Ed},
    booktitle = {Proceedings of the Conference on Design, Automation and Test in Europe (DATE'16)},
    year = {2016},
    month = {March},
    url = {https://www.researchgate.net/publication/304010295_Collective_Knowledge_Towards_RD_Sustainability}
}

@inproceedings{Fur2009,
  author =    {Grigori Fursin},
  title =     {{Collective Tuning Initiative}: automating and accelerating development and optimization of computing systems},
  booktitle = {Proceedings of the GCC Developers' Summit},
  year =      {2009},
  month =     {June},
  location =  {Montreal, Canada},
  keys =      {http://www.gccsummit.org/2009}
  url  =      {https://scholar.google.com/citations?view_op=view_citation&hl=en&user=IwcnpkwAAAAJ&cstart=20&citation_for_view=IwcnpkwAAAAJ:8k81kl-MbHgC}
}
```

* http://arxiv.org/abs/1506.06256
* http://hal.inria.fr/hal-01054763
* https://hal.inria.fr/inria-00436029
* http://arxiv.org/abs/1407.4075
* https://scholar.google.com/citations?view_op=view_citation&hl=en&user=IwcnpkwAAAAJ&citation_for_view=IwcnpkwAAAAJ:LkGwnXOMwfcC

Feedback
========

If you have problems, questions or suggestions, do not hesitate to get in touch
via the following mailing lists:
* https://groups.google.com/forum/#!forum/collective-knowledge
* https://groups.google.com/forum/#!forum/ctuning-discussions

![logo](https://github.com/ctuning/ck-guide-images/blob/master/logo-validated-by-the-community-simple.png)
