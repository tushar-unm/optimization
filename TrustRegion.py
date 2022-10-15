from scipy import *
from Useful import *
import math
import numpy as np

def bisection(fobj, a, b, xtol=9.9e-13, maxiter=1000):
	"""
	Find the root using the bisection method.  I copied part of the
	function signature from scipy.optimize.bisect(), which is 
	scipy's version of this function

	Inputs:
		fobj:  object with member f, a one dimensional function taking a number

		a:  Number, one end of the bracketing interval

		b:  Number, other end of the bracketing interval

		xtol: Number, allowable tolerance on f for solution
			should be $\geq$ 0

		maxiter: Number, maximum of iterations allowable

	Returns:
		The found solution, or NaN if no solution was found
	"""
	iter_count = 0
	while True:
		x_mid = (b + a) / 2.0
		f_x_mid = fobj.f(x_mid)
		iter_count += 1
		if sign(f_x_mid) != sign(fobj.f(a)):
			b = x_mid
		elif sign(f_x_mid) != sign(fobj.f(b)):
			a = x_mid
		else:
			return NaN
		maxiter -= 1
		if math.fabs(f_x_mid) < xtol or maxiter == 0:
			break
		
	if maxiter == 0:
		x_mid = NaN	
	
	return x_mid

class TauMinimizer:
	"""
	A small class that defines a function $f(\tau)$ with
	
	$f(\tau) = (p_j + \tau d_j)^T(p_j + \tau d_j) - \Delta^2$
	"""
	def __init__(self, p_j, d_j, Delta):
		"""
		Here in the constructor we define the values for
		$p_j$, $d_j$, and $\Delta$ for use in our function.
		"""
		self.p_j = p_j
		self.d_j = d_j
		self.Delta = Delta
	
	def f(self, tau):
		"""
		Return the function with the parameters given in the ctor
		evaluated at tau = $\tau$
		"""
		return dot(self.p_j + tau*self.d_j, self.p_j + tau*self.d_j) - self.Delta**2
			

def CG_Steihaug(fobj, epsilon, Delta, B_k, g):
	"""
	Note that fobj.n should be set to something useful
	before function entry.  Calculates the $p_k$ for the
	next iteration.  Based on algorithm 4.3 on page 75 
	of Nocedal and Wright.
	
	Inputs:
		fobj is the function object, used mostly for n

		epsilon is the tolerance

		Delta is the trust region size

		B_k is the $B_k$, usually the exact hessian at $x_k$

		g is the gradient of f evaluated at $x_k$

	Outputs:
		returns a tuple with the new $p_k$ and 4 integer values, 
		3 of which will be zero and one of which will be one.  They 
		are in order:
		tolerance - meaning that the p_j matches the tolernace and is good
		boundary - meaning that p_j was outside our trust region
		negative - meaning that the algorithm encountered negative curvature
		maxiter - meaning that the looped maxed out at n
	
	Side Effects:
		None
	"""
	# make a $p_j$ with dimension of the fobj
	# set $p_0 = p_j = p_0$
	p_j = zeros((fobj.n))
	# set $r_0 = r_j = g$
	r_j = g
	# set $d_0 = d_j = -r_0$
	d_j = -r_j

	#initialize dimensionality and storage in terms of n (the dimensionality of the problem)
	n = fobj.n
	# so (len(x_k) + 2 floating values for p_j + 2 floating value for g + 2 for r_j + 2 for d_j + 1 floating value for iteration count + for delta + n*len(B_k))/n;
	# the division by n is to present thee result in terms of dimensionality of the problem
	storage = (2 + 2 + 2 + 2 + 1 + 1 + n*len(B_k))/n

	# if $\|r_0\| < \epsilon$
	mag_r_0 = math.sqrt(dot(r_j, r_j))
	if mag_r_0 < epsilon:
		# lucked out on choice of p_0
		return (p_j, 1, 0, 0, 0)
	
	#initialize reporting dict to capture runtime parameters
	rep_dict = {}

	# for $j = 0, 1, 2, \ldots$ dimension of $B$
	for j in range(0, 10000):
		print('\n begin iteration {}'.format(j))
		iter_rep_dict = {}
		iter_rep_dict['p_j'] = p_j
		iter_rep_dict['p_j_plus_1'] = None
		iter_rep_dict['alpha_k'] = None
		iter_rep_dict['grad_f_x_k'] = r_j
		iter_rep_dict['grad_f_x_k_plus_1'] = None
		iter_rep_dict['storage'] = storage
		iter_rep_dict['f_eval'] = fobj.eval_count()[0]
		iter_rep_dict['grad_f_eval'] = fobj.eval_count()[1]
		iter_rep_dict['hessian_f_eval'] = fobj.eval_count()[2]
		iter_rep_dict['match_tolerance'] = None
		iter_rep_dict['outside_boundary'] = None
		iter_rep_dict['neg_curvature'] = None
		iter_rep_dict['max_iter'] = None
		rep_dict[j] = iter_rep_dict



		# if $d_j^TB_kd_j \leq 0$
		if MatQuad(d_j, B_k, d_j) <= 0:
			# encountered negative curvature
			# Find $\tau$ such that $p = p_j + \tau d_j$ minimizes
			# $m(p)$ in (4.9) and satisfies $\|p\| = \Delta$
			tm = TauMinimizer(p_j, d_j, Delta)
			tau_1 = bisection(tm, 0.0, 200.0*Delta)
			tau_2 = bisection(tm, 0.0, -200.0*Delta)
			p_1 = p_j + tau_1 * d_j
			p_2 = p_j + tau_2 * d_j
			iter_rep_dict['neg_curvature'] = 1
			if (np.linalg.norm(p_1,2) < np.linalg.norm(p_2,2)):
				iter_rep_dict['p_j'] = p_1
				rep_dict[j] = iter_rep_dict
				print(iter_rep_dict)
				return rep_dict
			else:
				iter_rep_dict['p_j'] = p_2
				rep_dict[j] = iter_rep_dict
				print(iter_rep_dict)
				return rep_dict

		# $\alpha_j = r_j^Tr_j / d_j^TB_kd_j$
		alpha_j = dot(r_j, r_j) / MatQuad(d_j, B_k, d_j)
		# $p_{j+1} = p_j + \alpha_jd_j$
		p_j_plus_1 = p_j + alpha_j * d_j

		iter_rep_dict['alpha_k'] = alpha_j
		iter_rep_dict['p_j_plus_1'] = p_j_plus_1

		# if $\|p_{j+1}\| \geq \Delta$
		if math.sqrt(dot(p_j_plus_1, p_j_plus_1)) >= Delta:
			# reached trust region boundary
			tm = TauMinimizer(p_j, d_j, Delta)
			tau_1 = bisection(tm, 0.0, 200.0*Delta)
			tau_2 = bisection(tm, 0.0, -200.0*Delta)
			iter_rep_dict['outside_boundary'] = 1
			if tau_1 > 0:
				iter_rep_dict['p_j'] = p_j + tau_1*d_j
				rep_dict[j] = iter_rep_dict
				print(iter_rep_dict)
				return rep_dict
			else:
				iter_rep_dict['p_j'] = p_j + tau_2*d_j
				rep_dict[j] = iter_rep_dict
				print(iter_rep_dict)
				return rep_dict

		# $r_{j+1} = r_j + \alpha_jB_kd_j$
		r_j_plus_1 = r_j + alpha_j*np.matmul(B_k, d_j)
		iter_rep_dict['grad_f_x_k_plus_1'] = r_j_plus_1

		# if $\|r_{j+1}\| < \epsilon\|r_0\|$
		if math.sqrt(dot(r_j_plus_1, r_j_plus_1)) < epsilon*mag_r_0:
			#print 'met stopping test'
			iter_rep_dict['match_tolerance'] = 1
			iter_rep_dict['p_j'] = p_j_plus_1
			rep_dict[j] = iter_rep_dict
			print(iter_rep_dict)
			return rep_dict

		# $\beta_{j+1} = r_{j+1}^Tr_{j+1}/r_j^Tr_j$
		beta_j_plus_1 = dot(r_j_plus_1, r_j_plus_1) / dot(r_j, r_j)

		# $j_{j+1} = r_{j+1} + \beta_{j+1}d_j$
		d_j_plus_1 = -r_j_plus_1 + beta_j_plus_1*d_j

		# advance all the plus ones
		p_j = p_j_plus_1
		r_j = r_j_plus_1
		d_j = d_j_plus_1

		for ky, v in iter_rep_dict.items():
			print('{}: {}'.format(ky, v))
		print('\n end iteration {} \n************\n**************\n'.format(j))

		rep_dict[j] = iter_rep_dict

	# if we get here we maxed out of the loop
	# maxed out on loop
	return rep_dict
			
			
def localmodel(fobj, x_k, p):
	"""
	Returns 

	
		$f(x_k) + \nabla f(x_k)^T p + \frac{1}{2}p^T\nabla^2f(x_k) p$
		
		with $B$ the exact hessian
	"""
	return fobj.f(x_k) + dot(fobj.grad_f(x_k), p) \
		+ 0.5*MatQuad(p, fobj.hessian_f(x_k), p)
			
def TrustRegion(x_0, Delta_Bar, Delta_0, eta, fobj, maxiter):
	"""
	This is an implementation of algorithm 4.1 in Nocedal
	and Wright, page 68.
	
	Inputs:
		x_0 is the $x_0$ value, an initial value for $x$
		
		Delta_Bar is the $\overline{\Delta}$ value, 
			the max value for Delta
		
		Delta_0 is the $\Delta_0$ value, the inital radius
			of the trust region
			
		eta is the $\eta$ value, the minimum acceptable
			reduction of $f_k$
		
		fobj is the function object
		
		maxiter is the maximum number of iterations accepted.
	
	Outputs:
		Returns the stationary point of $f$, $x^\ast$.
	
	Side Effects:
		Prints the counts of the different return states of
		CG_Steihaug at the end of the algorithm.
	"""
	x_k = x_0
	Delta_k = Delta_0

	# we keep counters of the exit conditions of CG_Steihaug
	tol = 0
	bound = 0
	neg = 0
	maxed = 0

	for k in range(0, maxiter):
		p_k, t, b, n, m = CG_Steihaug(fobj, 9.9e-13, Delta_k, fobj.hessian_f(x_k), fobj.grad_f(x_k))

		# update the exit condition counters
		tol += t
		bound += b
		neg += n
		maxed += m

		# $\rho_k = \frac{f(x_k) - f(x_k + p_k)}{m_k(0) - m_k(p_k)}$
		rho_k = (fobj.f(x_k) - fobj.f(x_k + p_k)) \
			 / (localmodel(fobj, x_k, zeros((fobj.n))) \
			    - localmodel(fobj, x_k, p_k))
		 
		# precaculate $\|p_k\|$
		mag_p_k = math.sqrt(dot(p_k, p_k))

		# if $\rho_k < \frac{1}{4}$
		if rho_k < 0.25:
			# $\Delta_{k+1} = \frac{1}{4}\|p_k\|$
			Delta_k = 0.25*mag_p_k
		else:
			# if $\rho_k > \frac{3}{4}$ and $\|p_k\| = \Delta_k$
			if rho_k > 0.75 and mag_p_k - Delta_k < 9.9e-13:
				Delta_k = 2*Delta_k
				if Delta_k > Delta_Bar:
					Delta_k = Delta_Bar
		# if $\rho_k > \eta$
		if rho_k > eta:
			# $x_{k+1} = x_k + p_k$
			x_k = x_k + p_k
		
		# break out if we've found a stationary point within tolerance
		if math.sqrt(dot(fobj.grad_f(x_k), fobj.grad_f(x_k))) < 9.9e-13:
			break

	print('tolerance met: %d, trust region boundary: %d, negative curvature: %d, maximum iterations: %d' % \
	    (tol, bound, neg, maxed))

	print('Number of function evaluations: %d' % fobj.eval_count())

	return x_k
		
		
#####################
# unit testing code #
#####################

import unittest
import ObjectiveFunctions

class TrustRegionTestCase(unittest.TestCase):
	def testTrustRegion(self):
		x_0 = array((stats.rand.uniform(-100.0, 100.0)[0], \
			stats.rand.uniform(-100.0, 100.0)[0]))
		alpha_0 = 1.0
		f1 = ObjectiveFunctions.testObjFunc()
		
		x_star = TrustRegion(x_0, 10.0, 1.0, 0.9, f1, 1000)
		
		assert alltrue(x_star == array((0.0, 0.0)))

		x_0 = array((stats.rand.uniform(-100.0, 100.0)[0], \
			stats.rand.uniform(-100.0, 100.0)[0]))
		alpha_0 = 1.0
		f2 = ObjectiveFunctions.testSecondFunc()

		x_star = TrustRegion(x_0, 10.0, 1.0, 0.9, f2, 1000)

		assert alltrue(x_star == array((1.0, 1.0)))


if __name__ == '__main__':
    unittest.main()

