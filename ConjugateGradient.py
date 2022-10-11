from scipy import *
from Useful import *

def MakeHilbertMatrix(n):
	"""
	Makes a Hilbert matrix of dimension nxn.  The $ij$th component
	of the Hilbert matrix is defined to be:
		
	$A_{i,j} = 1/(i+j-1)$
	
	Inputs:
		n is the dimension of the desired Hilbert matrix
	
	Outputs:
		Returns a Hilbert matrix of dimension nxn
	
	Side Effects:
		None
	"""
	hilbert = []
	
	for i in range(0, n):
		hilbert.append([])
		for j in range(0, n):
			# this is $i + j + 1$ because these array 
			# indicies are off of matrix indicies by -1,
			# in other words the array indicies are from
			# $[0..n-1]$ and the matrix indicies are from
			# $[1..n]$.  Therefore, we have to add 1 to
			# both $i$ and $j$, giving $((i+1) + (j+1) - 1)$
			# which equals $i + j + 1$.
			hilbert[i].append(1.0 / (i + j + 1))
		
	return array((hilbert))


def ConjugateGradient(x_0, A, b, residual, maxiter):
	"""
	This is an implementation of Algorithm 5.2 defined
	on page 111 of Nocedal and Wright.  It provides an
	estimate for the inverse of a matrix A to approximately
	solve the linear system
	
	$Ax = b$
	
	With A an nxn array, x a vector of dimension n, and
	b a solution vector of dimension n.
	
	Inputs:
		x_0, an inital estimate of the solution $x^\ast$
		
		A the nxn matrix for which we need an inverse to
			solve the linear system
		
		b the n dimension vector
	
		residual is the tolerance allowed on $r_k^Tr_k$
		
		maxiter is the maximum number of iterations allowable
	
	Outputs:
		Returns the number of iterations required to reduce
		the error factor $r_k^Tr_k$ below the residual argument

		Also the value $x^\ast$ that it found
	"""
	# $r_0 = r_k \leftarrow Ax_0 - b$
	r_k = matrixmultiply(A, x_0) - b
	# $p_0 = p_k \leftarrow -r_0$
	p_k = -r_k
	# $k \leftarrow 0$
	k = 0		
	x_k = x_0
	
	while math.sqrt(dot(r_k, r_k)) > residual and k < maxiter:
		# $\alpha_k \leftarrow \frac{r_k^Tr_k}{p_k^TAp_k}$
		alpha_k = dot(r_k, r_k) / MatQuad(p_k, A, p_k)
		
		# $x_{k+1} \leftarrow x_k + \alpha_kp_k$
		x_k = x_k + alpha_k*p_k
		
		# $r_{k+1} \leftarrow r_k + \alpha_kAp_k$
		r_k_plus_1 = r_k + alpha_k*matrixmultiply(A, p_k)
		
		# $\beta_{k+1} \leftarrow \frac{r_{k+1}^Tr_{k+1}}{r_k^Tr_k}$
		beta_k_plus_1 = dot(r_k_plus_1, r_k_plus_1) / dot(r_k, r_k)
		
		# $p_{k+1} \leftarrow -r_{k+1} + \beta_{k+1}p_k$
		p_k = -1.0*r_k_plus_1 + beta_k_plus_1*p_k
		
		# $k \leftarrow k+1$
		k += 1
		
		# advance $r_k$
		r_k = r_k_plus_1
		
	return (k, x_k)

def GetCGIterations(n):
	"""
	Returns the number of iterations required to solve the
	linear system $Ax = b$ of dimension $n$ using the
	ConjugateGradient algorithm defined above, and with
	$A$ as an nxn Hilbert matrix.
	
	Inputs:
		n, the dimension of the system
		
	Outputs:
		The number of iterations CG required to solve the
		system
		
	Side Effects:
		None
	"""
	A = MakeHilbertMatrix(n)
	b = ones(n)
	x_0 = zeros(n)
	iters, x_star = ConjugateGradient(x_0, A, b, 10e-6, 10000)
	return iters

#####################
# unit testing code #
#####################

import unittest

class ConjugateGradientTestCase(unittest.TestCase):
	def testConjugateGradient(self):
		# CG should solve identity system in one iteration
		A = identity(300)
		
		x_0 = zeros(300)
		b = ones(300)
		
		(iters, x_star) = ConjugateGradient(x_0, A, b, 0, 300)
		
		assert iters == 1

		remains = matrixmultiply(A, x_star) - b

		assert dot(remains, remains) == 0
		
		
if __name__ == '__main__':
    unittest.main()

