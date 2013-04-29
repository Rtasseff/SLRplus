from SLRplus import SLRplusModel

def simpFull(runId,X,y):
	"""Runs a complete SLRplus work flow including:
	full cv fit, p value calc and impact value calc.
	The standard defualts are used.
	The results are stored in a text file 
	'SLRrun_'+runId+'.dat' in the woriking dir.
	"""
	slr = SLRplusModel(X,y) 
	slr.calcFit()
	slr.calcPValues()
	slr.calcImpactValues()
	path = 'SLRrun_'+runId+'.dat'
	slr.save(path)


