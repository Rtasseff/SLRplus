from dispatch import dispatcher, smp
import numpy as np
# import the run modual
import runSLR

# load the regressor matrix
#X = np.loadtxt('X.dat',delimiter="\t")
#Y = np.loadtxt('Y.dat',delimiter="\t")

#X = np.loadtxt('th.dat')
#Y = np.loadtxt('dth.dat')
X = np.loadtxt('x.dat')
Y = np.loadtxt('dx.dat')


n,m = Y.shape

disp = smp.SMPDispatcher(m)
for i in [1895]:#for i in range(m):
	y = Y[:,i]
	#Xhat = np.sin(X.T-X[:,i]).T
	Xhat = X
	args = (str(i),Xhat,y)
	job = dispatcher.Job(runSLR.simpFull,args)
	disp.add_job(job)

disp.dispatch()
