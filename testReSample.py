#simple function to run a test on resampling stratagies
# 250 samples on each
#uses bootstrap full
#bootstrap selection bs residual full
#bootstrap selection bs residual selected
#bootstrap selection boostrap selected
import elasticNetLinReg as enet
from glmnet import glmnet
import numpy as np
import cvTools as st

def runTestPrint(X,y,name):
	vals = runTest(X,y)
	np.savetxt(name+'.dat',vals)

def runTest(X,y):
	nSamp = 250
	nObs,nRegs = X.shape
	
	# selection via bootstrap
	err,enm,allVals = fitSampling(X,y,1,nSamp,method='bs')
	errV = err.mErr
	tmpIndex = np.argmin(errV)

	# get the bootstrap full values
	bsAll = allVals[tmpIndex,:]

	# other important values
	lam = enm.lambdas[tmpIndex]
	yHat = enm.predict(X)[:,tmpIndex]
	coefIndex = enm.indices
	
	# get the bootstrap residual response samples
	res = y - yHat
	resCent = res-np.mean(res)
	ySample = np.zeros((nObs,nSamp))
	for i in range(nSamp):
		resSample = st.sampleWR(resCent)
		ySample[:,i] = yHat+resSample

	# get the cv error estimated over bs residual responses
	errSample = np.zeros(nSamp)
	for i in range(nSamp):
		err,tmp,tmpallVals = fitSampling(X,ySample[:,i],1,10,method='cv',lambdas=[lam])
		errV = err.mErr
		#should be only one value here
		if len(errV)>1:
			raise ValueError('something wrong with bs res cv')
		
		errSample[i] = errV[0]
		
	bsResAll = errSample

	#now let repeat this stuff on the selected sample
	
	Xhat = X[:,coefIndex]
	err,enm,allVals = fitSampling(Xhat,y,1,nSamp,method='bs',lambdas=[lam])
	
	bsSub = allVals

	# get the cv error estimated over bs residual responses
	errSample = np.zeros(nSamp)
	for i in range(nSamp):
		err,tmpenm,tmpallVals = fitSampling(Xhat,ySample[:,i],1,10,method='cv',lambdas=[lam])
		errV = err.mErr
		#should be only one value here
		if len(errV)>1:
			raise ValueError('something wrong with bs res cv')
		
		errSample[i] = errV[0]
		
	bsResSub = errSample

	vals = np.zeros((4,nSamp))
	vals[0,:] = bsAll
	vals[1,:] = bsResAll
	vals[2,:] = bsSub
	vals[3,:] = bsResSub

	return vals
			
def calcDeviation(x):
	xMean = np.mean(x)
	n = len(x)
	dev = np.zeros(n)
	for i in range(n):
		dev[i] = (np.mean(x[:(i+1)])-xMean)/xMean

	return dev
		

	


def fitSampling(regressors, response, alpha, nSamp, method='cv', 
		memlimit=None, largest=None, **kwargs):
	"""Performs an elastic net constrained linear regression,
	see fit, with selected sampleing method to estimate errors
	using nSamp number of sampleings.
	methods:
	'cv'	cross validation with nSamp number of folds
	'bs'	bootstrap 
	'bs632'	boostrap 632 (weighted average of bs and training error)
	Returns a TrainingError object (cvTools) and an 
	ENetModel object for the full fit (err,enm).
	Function requires cvTools
	"""
	
	nObs,nRegs = regressors.shape
	# get the full model fit 
	fullEnm = enet.fit(regressors, response, alpha, memlimit,
                largest, **kwargs)
	# get the lambda values determined in the full fit (going to force these lambdas for all cv's)
	lam = fullEnm.lambdas
	# the lambdas may have been user defined, don't want it defined twice 
	if kwargs.has_key('lambdas'):
		del kwargs['lambdas']

	# lets partition the data via our sampling method
	if method=='cv':
		t,v = st.kFoldCV(range(nObs),nSamp,randomise=True)
	elif (method=='bs') or (method=='bs632'):
		t,v = st.kRoundBS(range(nObs),nSamp)
	else:
		raise ValueError('Sampling method not correct')

	# lets consider many versions of errors
	# with our error being mean squared error
	# we want the epected mean squared error
	# and the corisponding variance over the diffrent versions
	nModels = len(lam)
	smse = np.zeros(nModels)
	sSqmse = np.zeros(nModels)
	allVals = np.zeros((nModels,nSamp))

	# *** track the coefficent values as well
	# since spasitry can change (coef can be exactly zero in 
	# some folds but not others) we are tracking all of them
	# not good for memory
	sc = np.zeros((nRegs,nModels))
	sSqc = np.zeros((nRegs,nModels))
		# loop through the folds
	for i in range(nSamp):
		# get the training values
		X = regressors[t[i]]
		y = response[t[i]]
		enm =  enet.fit(X, y, alpha, memlimit,
                	largest, lambdas=lam, **kwargs)
		# coef time
		sc[enm.indices,:] = sc[enm.indices,:] + enm.coef
		sSqc[enm.indices,:] = sSqc[enm.indices,:] + enm.coef**2
		# get the validation values
		Xval = regressors[v[i]]
		Yval = response[v[i]]
		nVal = float(len(Yval))
		# get the predicted responses from validation regressors
		Yhat = enm.predict(Xval)
				# what is the mean squared error?
		# notice the T was necassary to do the subtraction
		# the rows are the models and the cols are the observations
		mse = np.sum((Yhat.T-Yval)**2,1)/nVal
		# sum the rows (errors for given model)
		smse = smse + mse
		sSqmse = sSqmse + mse**2
		allVals[:,i] = mse
		
	# now it is time to average and send back
	# I am putting the errors in a container 
	nSampFlt = float(nSamp)
	meanmse = smse/nSampFlt
	varmse = sSqmse/nSampFlt - meanmse**2
	if method=='bs632':
		yhat = fullEnm.predict(regressors)
		resubmse = np.sum((yhat.T-response)**2,1)/float(nObs)
		meanmse = 0.632*meanmse+(1-0.632)*resubmse
		
	mc = sc/nSampFlt
	vc = sSqc/nSampFlt - mc**2
	err = enet.ENetTrainError(lam,nSamp,meanmse,varmse,mc,vc,alpha)
	err.setParamName('lambda')

	fullEnm.setErrors(err.mErr)
	
	return err, fullEnm, allVals 

