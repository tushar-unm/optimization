from scipy import *

def MatQuad(l, M, r):
	"""
	In several instances the product $l^TMr$ appears
	and so we define it here to simplify the 
	implementation of various procedures.
	
	Inputs:
		l is an n-dimensional vector
		M is an nxn matrix
		r is an n-dimensional vector
	
	Outputs:
		returns the product $l^TMr$
	"""
	return matrixmultiply(matrixmultiply(transpose(l), M), r)