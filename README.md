SLRPlus - Tools for Sparse Linear Regression 
==================================
20111212 RAT

This is a fairly straight forward set of tools for 
running linear regression with constraints for stability and 
selection. Currently the only formulation considered is the 
elastic net formulation (which is a combination of ridge 
regression and LASSO, L2 and L1 norms).  The central algorithm
is coordinate descent as described in Friedman, Hastie and Tibshirani
J Stat Softw. 2010; 33(1): 1‚Äì22.

Note:  work in progress!!
Also I am not a programmer and I am not a long term 
python user, so be aware, organization and nomenclature
and anything else may be a little messy. 

Functionality
-------------
The primary calculations for the fit are done using the 
fortran code by Jerome Friedman, this is integrated via 
the python wrapper from: https://github.com/dwf/glmnet-python.

As for my tiny contribution I wrote up a set of tools to preform 
some common tasks that I find useful. The two most relevant modules 
are elasticNetLinReg.py and SLRplus.py.  

elasticNetReg.py has methods to fit a model (I typically find the 
default properties to be pretty useful but they can easily be changed),
run cv analysis for multiple lambdas (penalty parameter) and construct objects
to represent the results for efficient access of important properties
and plotting.

basic example:

    In [50]: import numpy as np
    In [51]: import elasticNetLinReg as enet
    In [52]: X = np.random.randn(20,100)
    In [53]: w = 5*np.random.randn(100)
    In [54]: w[5:] = 0
    In [55]: y = np.dot(X,w)
    In [56]: enm = enet.fit(X,y,1)
    In [60]: enm.indices
    Out[60]: array([ 3, 39, 15,  2, 51, 42,  0, 69,  9, 63,  4, 53,  5,  1])
 
enm is a model object of the fit of the data with an alpha (balance 
parameter) of 1, in this example, equivalent to standard lasso.  The coordinate decent  
algorithm naturally solves for a trajectory of lambda values so enm has
results for many fits.  The indices displayed is a sparse representation
of the regressors that were assigned non-zero coefficients for any of the
fits considered.

SLRplus.py has a class to run and store the regression as 
well as a few other properties that may be useful.  The regression
parameters (lambda and alpha) are both chosen by cross validation,
There is a function to estimate the significance (p values) that the 
coef are not random, and a function to estimate the impact of removing
selected regressors.  

basic example:

    In [50]: import numpy as np
    In [51]: import elasticNetLinReg as enet
    In [52]: X = np.random.randn(20,100)
    In [53]: w = 5*np.random.randn(100)
    In [54]: w[5:] = 0
    In [55]: y = np.dot(X,w)
    In [61]: import SLRplus
    In [62]: slr = SLRplus.SLRplusModel(X,y)
    In [63]: slr.calcFit()
    In [64]: slr.calcPValues()
    In [65]: slr.calcImpactValues()
    In [66]: slr.indices
    Out[66]: array([ 3, 39, 15,  2, 51,  0, 69,  9, 63,  4, 53,  5,  1])
    In [67]: slr.pValues
    Out[67]: 
    array([ 0.   ,  0.99 ,  0.504,  0.   ,  0.446,  0.   ,  0.387,  0.813,
        0.814,  0.   ,  0.446,  0.666,  0.   ])
   

Dependencies
------------
I used python 2.7, numpy 1.6.1, and scipy 0.10.0.
Matplotlib 1.1.0 was used for the plotting methods.

IMPORTANT
You need the wrapper and the original fortran code for glmnet to do anything useful.
The code is maintained at: https://github.com/dwf/glmnet-python.
There is a little trick in the compile step so read the readme for that module.

To improve efficiency of p-value calculations I used an analysis (and code) from:
http://informatics.systemsbiology.net/EPEPT/.  The original code was in matlab,
and I converted it to python around 20111205.  It may be included here.

To allow for multiprocessing of jobs I used code (dispatcher/) from a colleague (JR) 
which may be included, this only impacts the files runSLR.py and runMP.py and not
the above functionality.



License
-------

*There is a license for the wrapper and the original fortran code but it is not
my intention to distribute or maintain that code so I will leave you to find it 
when and if you decide to download it (note: it is freely available).

*For the gpdPerm module based on source code at http://informatics.systemsbiology.net/EPEPT/:
Copyright (C) 2003-2010 Institute for Systems Biology, Seattle, Washington, USA.
 
The Institute for Systems Biology and the authors make no representation about the suitability or accuracy of this software for any purpose, and makes no warranties, either express or implied, including merchantability and fitness for a particular purpose or that the use of this software will not infringe any third party patents, copyrights, trademarks, or other rights. The software is provided "as is". The Institute for Systems Biology and the authors disclaim any liability stemming from the use of this software. This software is provided to enhance knowledge and encourage progress in the scientific community. 
 
This is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.
 
You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
http://www.gnu.org/licenses/lgpl.html
