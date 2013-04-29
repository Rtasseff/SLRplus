
edgeOut = open('edge.prop','w')
nodeOut = open('node.prop','w')
errOut = open('missed.txt','w')

nodeOut.write('node id'+'\t'+'lambda'+'\t'+'alpha'+'\t'+'variance'+'\t'+'error'+'\t'+'intercept'+'\n')
edgeOut.write('source id'+'\t'+'target id'+'\t'+'coef'+'\t'+'meanCoef'+'\t'+'varCoef'+'\t'+'pValues'+'\t'+'impact'+'\n')

nRuns = 3#3078

for i in range(nRuns):
	# assuming run number is node number
	inNode = str(i)
	try:
		# assuming standard name and in this dir
		path = 'SLRrun_'+inNode+'.dat' 
		f = open(path,'r')
		# get node properties 
		line = f.next()
		lam = line.split()[0]

		line = f.next()
		alpha = line.split()[0]

		line = f.next()
		varResp = line.split()[0]

		line = f.next()
		error = line.split()[0]

		line = f.next()
		intercept = line.split()[0]
		
		
		# get edge properties

		line = f.next()
		indices = line.split()

		line = f.next()
		coefs = line.split()

		line = f.next()
		meanCoef = line.split()

		line = f.next()
		varCoef = line.split()

		line = f.next()
		pVal = line.split()

		line = f.next()
		iVal = line.split()
		

		# having me here will mean if there is any mistake in reading node wont exist at all
		# missing edges should be seen as empty sets, not errors and will still have nodes
		# just no edges.
		nodeOut.write(inNode+'\t'+lam+'\t'+alpha+'\t'+varResp+'\t'+error+'\t'+intercept+'\n')
		n = len(indices)

		for j in range(n):
			edgeOut.write(indices[j]+'\t'+inNode+'\t'+coefs[j]+'\t'+ \
				meanCoef[j]+'\t'+varCoef[j]+'\t'+pVal[j]+'\t'+iVal[j]+'\n')
		
		f.close()
	except:
		errOut.write(inNode+'\n')	

errOut.close()
nodeOut.close()
edgeOut.close()
	
