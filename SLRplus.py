##########
# Created 20111206 RAT
# scripts to run all the analysis of a
# sparse linear regression 
#########
import numpy as np
import elasticNetLinReg as enet


class SLRplusModel(object):
	"""Creates and stores a sparse linear regression model
	with functions to estimate vairious properties of the 
	coefficents.  Currently uses elastic net regression, 
	both the weight (labda) and balance (alpha) parrameters
	are selected via cross validation.  Regression is performed 
	on command, after initialization. Due to computational costs
	some properties are estimated on command after regression.
	""" 
	# the original elastic net object i wrote is for limited info on many lambda values
	# here we are considering only a single model
	# I first tried extending elasticNetLinReg.ENetModel
	# but basiclly would have had to over written everything for
	# my purposes here

	def __init__(self,regressors,response):
		"""Initializes the SLR model
		regressors	matrix of predictor values
				cols=regressors,rows=obs
		response	vector of responses at obs
		"""
		self._X = regressors
		self._y = response
		self._pValsCalcd = False
		self._iValsCalcd = False
		self._fitCalcd = False

	def calcFit(self,alpha=np.arange(.1,1.1,.1),nSamp=10,sampling='cv'):
		"""Calculates the regression using cross validation 
		to determine the parameters.  Calls 
		elasticNetLinReg.fitFullCV.
		alpha	eNet param to test for
		nSamp	number of rounds for sampling
		sampling	sampling method, cross validation ('cv')
				bootstrap ('bs'), bootstrap .632 ('bs632')
				see elasticNetLinReg.fitSampling 
		return: void
		sets most properties of the model object including coef estimates 
		"""
		# may need these latter
		self._alphaRange = alpha
		self._nSamp = nSamp	
		self._sampling = sampling	

		# run the full cv fit
		# I have found that doing this iterativly while removing unsleceted varriables is better
		# *** hack, having the function passing back nullErr as a single value
		enm, mc,vc, nullErr = enet.fitFull(self._X,self._y,alpha,nSamp,sampling)
		self._nullErr = nullErr		
		# remember the elastic net object is typically for many models (lambda scanning)
		# fitFullCV should return a single model, but many properties will be
		# in arrays of higher dimessions then needed here
		# this seems kinda sloppy
		self._c = enm.coef
		self._i = enm.indices
		self._lam = np.float64(enm.lambdas[0]) # had probs w/usr defined values as float
		self._al = np.float64(enm.alpha) # had probs w/usr defined values as float
		self._c0 = enm.intercept[0]
		self._err = enm.errors[0]
		self._fitCalcd = True
		# the vectors from enet are for all possible regressors
		# lets sparseify that down a bit
		# *** analysis allert
		# the typical coef will be from the full fit across the data
		# however the cross validation can(will) have diffrent values at 
		# each fold, this means that some coef may be present in some 
		# folds and not in others, thus some non zero mean coef values
		# will be lost here (but I ask you, are they important???)
		self._mc = mc[enm.indices]
		self._vc = vc[enm.indices]

		# Ok we have a sparse rep of the non-zero regressors...
		# or do we??
		# the original glmnet function will add indices 
		# to the sparse list when they are needed, but will nerver remove them
		# from this list; even if the values are reset to zero latter in the fit.
		# It is possible that the model we choose has zero coef values in this list,
		# lets kill those now to save on computation latter!
		nonZero = np.abs(self._c)>1E-52
		self._i = self._i[nonZero]
		self._c = self._c[nonZero]
		self._mc = self._mc[nonZero]
		self._vc = self._vc[nonZero]
		
		
		
	
	
	@property
	def lam(self):
		# cannot call this lambda Ha that has a meaning I guess
		"""Cross validation selected lambda.
		The elastic net penalty weight parameter.
		"""
		if self._fitCalcd:
			return self._lam
		else:
			raise LookupError("The fit has not been calculated.")
	
	@property
	def varResp(self):
		"""Recently changed: not the varriance
		in the response rather the error if no 
		regressors used.  Very similar to response 
		variance, but resulting from the sampling
		(for huge samples this would be equal to var)
		"""
		# *** fix my name
		#"""The variance of the response vector"""
		return self._nullErr


	@property
	def coef(self):
		"""Vector of the regression coefficents
		from the full fit, values corrispond to 
		the regressors in self.indices.
		"""
		if self._fitCalcd:
			return self._c
		else:
			raise LookupError("The fit has not been calculated.")

	@property
	def indices(self):
		"""Vector of the regression indices
		from the full fit, values corrispond to 
		the columns of self.regressors.
		"""
		if self._fitCalcd:
			return self._i
		else:
			raise LookupError("The fit has not been calculated.")
		
			

	@property
	def alpha(self):
		"""Cross validation selected alpha.
		The elastic net balance parameter.
		"""
		if self._fitCalcd:
			return self._al
		else:
			raise LookupError("The fit has not been calculated.")

	@property
	def intercept(self):
		"""Scalar for the regression intercept."""
		if self._fitCalcd:
			return self._c0
		else:
			raise LookupError("The fit has not been calculated.")

	@property
	def error(self):
		"""The overall model error determined by cross validation.
		The defualt will be the mean squared error.
		"""
		if self._fitCalcd:
			return self._err
		else:
			raise LookupError("The fit has not been calculated.")
	
	@property
	def meanCoef(self):
		"""The mean value of the selected coefficents
		determined by the cross validation.  Values 
		corrispond to the regressors in self.indices		
		"""
		if self._fitCalcd:
			return self._mc
		else:
			raise LookupError("The fit has not been calculated.")	
		
	@property
	def varCoef(self):
		"""The mean varriance of the selected coefficents
		determined by the cross validation.  Values 
		corrispond to the regressors in self.indices		
		"""
		if self._fitCalcd:
			return self._vc
		else:
			raise LookupError("The fit has not been calculated.")	
		
				
	@property
	def pValues(self):
		"""P value vector for selected coefficents corrisponding to regressors 
		listed in self.indices.  The value is a measure of the coefficents 
		statistical significance, ie the probability that a coefficent's value may 
		have been random.
		
		set by calcPValues
		"""
		if self._pValsCalcd:
			return self._pVals
		else:
			raise LookupError("P values have not been calculated.")
				
	@property
	def impactValues(self):
		"""Impact value vector for selected coefficents corrisponding to regressors 
		listed in self.indices.  The value is a measure of the coefficents
		impact on the models pridictive power, ie error of model without 
		the corrisponding regressor.
		Results can be compared to self.error (error for complete model).
		
		set by calcImpactValues
		"""
		if self._iValsCalcd:
			return self._iVals
		else:
			raise LookupError("Impact values have not been calculated.")

	
	def predict(self,regressors,sparse=False,estVar=False):
		"""Retruns a vector of predicted responses
		given a matrix of regressors at diffrent observations.
		
		If sparse=False, passed regressor matrix is of same form as 
		self.regressors; else the col of passed regressor matrix
		should corrispond to self.indices.
		If estVar=False, self.coef (full fit) are used;
		else self.meanCoef are used for fit and 
		self.varCoef are used to estimate response variance
		due to the coef varriance (I think).
		returns vector of predicted responses (or if cv matrix with
		first col being the responses and the second the variance) 
		"""
        	regressors = np.atleast_2d(np.asarray(regressors))
		if not sparse: X = regressors[:,self.indices]
		
		if not estVar:
        		return self.intercept + np.dot(X, self.coef)
		else:
			n,m = X.shape
			sol = np.zeros((n,2))
			sol[:,0] = self.intercept + np.dot(X, self.meanCoef)
			sol[:,1] = self.intercept + np.dot(X, self.varCoef)	
			return sol
	

	  
	def calcPValues(self,nperm=1000,mode='net'):
		"""This method will calculate the p values for the 
		selected regressors.  The p value is the probability that
		the coef is from the null distribution (no corrilation to response).
		Currently the only methods avalible use:
		The test statistic is the t-statistic
		The null dist is approximated by random permutations of the response
		A generalized pereto dist is used to imporve the null estimate when
		possible and needed.
		nperm	number of permutations used to estimate null distribution
		mode	string - 'ols' calculates the tStatistic using 
			ordinary least squares, and 'net' uses the elastic net 
			formulation with the predetermined lambda and alpha parameters 
		return: void
		no values returned, method directly sets self.pValues
		Note: calculation of values is time consuming.
		Note: in my observation of toy test cases (no theoretical proof) i have noticed
		that 'ols' tends to be more sensitive (>fp but <fn)
		and 'nets' tends to be more selective (<fp but >fn)
		These cases were done with no corrilation in varriables
		regs = 4*obs
		For real cases with regs>>obs and likley many corrilations
		I am not able to determine sensitivity or selectivity; however I 
		did notice a diffrent trend in that regs were more lilkey 
		to be significant with net.
		"""
		
		# model needed to be run already
		if not self._fitCalcd: 
			raise RuntimeError("The fit has not been calculated, but is requiered!")
		
		
		nObs,m = self._X.shape
		nReg = len(self.indices)
		
		# only continue if there are values
		if nReg>0: 		
			# unless a defualt is changed in the solver
			# we need to include an intercept term
			if np.abs(self.intercept) > 1E-52: 
				X = np.ones((nObs,nReg+1))
				X[:,:-1] = self._X[:,self.indices]

			else:
				X = self._X[:,self.indices]

			# calling out to another method to simplify multiple options
			if mode=='ols':
				p = self._calcPValues1(X,nperm)
			elif mode=='net':
				p = self._calcPValues2(X,nperm)
			else:
				raise ValueError("No p value method %s" % mode)
			# if the intercept was included, ignore it now
			p = p[:nReg]

		else: p = np.array([])
					
		self._pVals = p
		self._pValsCalcd = True

	def calcImpactValues(self,full=False):
		"""This method will calculate the impact values for 
		the selected regressors.  The impact value is the 
		predictability associated with a regressor.
		Mean cross validation squared error of the model when 
		regressor is removed and the fit is repeated.
		Results could be compared to self.errors which by defult is 
		the cross validation mse for the full model. 

		full - boolean: False (defulat) then the same parameters
		(lambda and alpha)and the same regressors from the original 
		fit are used in the fitting process; True then the entier fitting
		process is redone on the original regressor matrix (parameters
		are re chosen via cv)
		return: void
		no values returned, method directly sets self.impactValues
		Note: calculation of values is time consuming (full is more so).
		"""

		# model needed to be run already
		if not self._fitCalcd: 
			raise RuntimeError("The fit has not been calculated, but is requiered!")

		# calling out to another method to simplify 
		# future possible selection of calculation method
		if full:
			iVals = self._calcImpactValues1()
		else:
			iVals = self._calcImpactValues2()
				
		self._iVals = iVals
		self._iValsCalcd = True
			

	def plot(self,i,varEst=False):
		"""Plot the raw response vs the ith raw
		regressor (corrisponding to self.indices). 
		"""
		if not self._fitCalcd: 
			raise RuntimeError("The fit has not been calculated, but is requiered!")

		import matplotlib
		import matplotlib.pyplot as plt
		
		plt.clf()
		interactive_state = plt.isinteractive()
		plt.plot(self._X[:,self.indices[i]],self._y,'o')
		plt.xlabel('regressor %i' % self.indices[i])
		plt.ylabel('response')
		plt.show()
		plt.interactive(interactive_state)
	

	def save(self,path):
		"""Saves the data in this model 
		to a file in a standard way.  
		Note: Will over write file without warning.

		self - SLRplusModel
		path - file path to write to
		
		format:
		lambda
		alpha
		varResp
		error
		intercept
		indices
		coef
		meanCoef
		varCoef
		pValues
		impactValues

		Note: the first five lines are single values
		the rest are arrays so values will be sep by \t 
		"""
		
		if not self._fitCalcd: 
			raise RuntimeError("The fit has not been calculated, nothing to save")
		

		f = open(path,'w')	

		self.lam.tofile(f,sep="\t")
		f.write("\n")

		self.alpha.tofile(f,sep="\t")
		f.write("\n")
		
		self.varResp.tofile(f,sep="\t")
		f.write("\n")

		self.error.tofile(f,sep="\t")
		f.write("\n")

		self.intercept.tofile(f,sep="\t")
		f.write("\n")

		self.indices.tofile(f,sep="\t")
		f.write("\n")

		self.coef.tofile(f,sep="\t")
		f.write("\n")
		
		self.meanCoef.tofile(f,sep="\t")
		f.write("\n")

		self.varCoef.tofile(f,sep="\t")
		f.write("\n")

		if self._pValsCalcd:
			self.pValues.tofile(f,sep="\t")
			f.write("\n")

		if self._iValsCalcd:
			self.impactValues.tofile(f,sep="\t")
			f.write("\n")

		f.close()


	def _calcPValues1(self,X,nperm=1000):
		# tested using an ordinary least squares fit to evaluate
		# the t-statistic, permutation testing is used for the null
		# distribution and a genral pareto distribution is used 
		# to estimate the tail of the permutation dist when possible.
		# The method called is olsStat.ttestPermute
		# nperm	number of permutations used to estimate null distribution
		# X is the typical regressor matrix
		# y is the typical response vector
		# nperm is the number of permutations to do
		import regStat	
		
		y = self._y
		p,tStat,tStatPerm,coef = regStat.olsTTestPermute(X,y,nperm)
		n,m = tStatPerm.shape
		# would like to check if any values are nan
		# this most likly means the gpd failed in goodness of fit for tail
		# will use direct permutation values as the estimate in that case 
		# *** some other form of automated checking might be good here
		for i in range(n):
			if np.isnan(p[i]):
				z = tStatPerm[i,:]
				tmp = np.sum(z>tStat[i]) 
				p[i] = float(tmp)/float(m)
		
		return p


	def _calcPValues2(self,X,nperm=1000):
		# tested using the generalized linear model elastic
		# net fit (with predetermined alpha and lambda parameters) to evaluate
		# the t-statistic, permutation testing is used for the null
		# distribution and a genral pareto distribution is used 
		# to estimate the tail of the permutation dist when possible.
		# The method called is olsStat.ttestPermute
		# nperm	number of permutations used to estimate null distribution
		# X is the typical regressor matrix
		# y is the typical response vector
		# nperm is the number of permutations to do
		import regStat	
		
		y = self._y
		p,tStat,tStatPerm,coef = regStat.netTTestPermute(X,y,self.alpha,self.lam,nperm)
		n,m = tStatPerm.shape
		# would like to check if any values are nan
		# this most likly means the gpd failed in goodness of fit for tail
		# will use direct permutation values as the estimate in that case 
		# *** some other form of automated checking might be good here
		for i in range(n):
			if np.isnan(p[i]):
				z = tStatPerm[i,:]
				tmp = np.sum(z>tStat[i]) 
				p[i] = float(tmp)/float(m)
		
		return p


	def _calcImpactValues1(self):
		#This method will calculate the impact values as described 
		#in self.impactValues.
		#The method called is elasticNetLinReg.fitFullCV.
		#reruns the entire fitting process on full regressors
		#indices  the selected coefs
		#X is the full reggressor matrix
		#y is the response vector
		#alpha	alpha values to consider
		#nSamp	number of folds in cross validation
		# returns the impact values (mse with coef removed)
		
		indices = self.indices
		X = self._X
		y = self._y
		alpha = self._alphaRange
		nSamp = self._nSamp
		
		# pre set the impact values at zero
		n = len(indices)
		iVals = np.zeros(n)
		for i in range(n):
			index = indices[i]
			# remove the coef
			Xhat = np.delete(X,index,axis=1)
			# rerun the fit
			tmp, a, b, c = enet.fitFull(Xhat,y,alpha,nSamp,self._sampling)
			# get the error, only one model so get the scalar value
			iVals[i] = tmp.errors[0]
		
		return iVals

	def _calcImpactValues2(self):
		#This method will calculate the impact values as described 
		#in self.impactValues.
		#The method called is elasticNetLinReg.fitCV.
		#will run the fit for this models parameters over
		#uses the same alpha and lambda values  as the original fit
		#indices  the selected coefs
		#X is the full reggressor matrix
		#y is the response vector
		#alpha	alpha values to consider
		#nSamp	number of folds in cross validation
		# returns the impact values (mse with coef removed)
		
		indices = self.indices
		X = self._X
		y = self._y
		nSamp = self._nSamp
		
		# pre set the impact values at zero
		n = len(indices)
		iVals = np.zeros(n)
		for i in range(n):
			# remove the coef
			Xhat = np.delete(X,i,axis=1)
			# rerun the fit
			tmpCV, tmpM = enet.fitSampling(Xhat,y,self.alpha,nSamp,self._sampling, lambdas=[self.lam])
			# get the error, only one model so get the scalar value
			iVals[i] = tmpM.errors[0]
		
		return iVals


