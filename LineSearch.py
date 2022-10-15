from scipy import linalg
from math import *
import pandas as pd 
import sys
from Useful import *
import numpy as np
import code as c

def BacktrackingAlpha(fobj, x_k, p_k, alpha_bar):
	"""
	Uses the backtracking procedure defined on page 41
	of Nocedal and Wright to find the $\alpha_k$ to
	be used for an iteration of a line search 
	algorithm.

	Inputs:
		fobj is a function object from ObjectiveFunctions.py
		
		x_k is the $x_k$ point that we are finding $\alpha$ in
		
		p_k is the $p_k$ search direction
		
		alpha_bar is the $\overline{\alpha}$, the initial value for $\alpha$
	
	Outputs:
		The $\alpha$ value to use for this iteration
	"""
	# $\rho$ and $c$ are constants that we set here
	rho = 0.5
	c = 1e-4

	# $\alpha \leftarrow \overline{\alpha}$
	alpha = alpha_bar
	# repeat until 
	# $f(x_k + \alpha p_k) \leq f(x_k) + c \alpha \nabla f_k^T p_k$ 
	while fobj.f(x_k + alpha * p_k) > fobj.f(x_k) + \
			c * alpha * np.matmul(\
		transpose(fobj.grad_f(x_k)), p_k):
		# $\alpha \leftarrow \rho \alpha$
		alpha = rho * alpha
		
	return alpha

def NewtonsMethodLineSearch(x_0, alpha_0, f_alpha_k, fobj, xtol):
	"""
	Finds a point within a small distance from a stationary point 
	of the function f using the basic line search algorithm defined
	in chapter 3 of Nocedal and Wright's Numerical Optimization, where

	$x_{k+1} = x_k + \alpha_k p_k$

	For Newton's Method, the $B_k = \nabla^2f(x_k)$, so 

	$p_k = -\nabla^2f(x_k)\nabla f(x_k)$

	This function supports arbitrary line search procedures
	to find the $\alpha_k$.  

	Inputs:
		x_0 is a Numeric array giving the value of $x_0$.

		alpha_0 is a floating point scalar giving the value of $\alpha_0$.

		f_alpha_k is a function reference, a function that takes an
			$x_k$, a $p_k$, and an $\overline{\alpha}$ and returns
			an $alpha_k$.  For reference see BacktrackingAlpha above.

		fobj is an ObjectiveFunction object.  The functions must take 
			inputs of the same degree as x_0.
	
		xtol is the tolerance on the stationary point.  The 
			algorithm will terminate when $\|\nabla f(x_k)\| \leq$ xtol.
	
	Outputs:
		Returns the computed value of the stationary point of $f$, $x^\ast$.

	Side Effects:
		Writes the values of $x_k$, $\alpha_k$, and $p_k$ at each iteration
		to stdout.
	"""
	x_k = x_0
	iterations = 0


	#initialize dimensionality and storage in terms of n (the dimensionality of the problem)
	n = fobj.n
	# so (len(x_k) + 2 floating values for pk + 1 floating value for alpha_k + 1 floating value for iteration count)/n; 
	# the division by n is to present thee result in terms of dimensionality of the problem
	storage = (len(x_k) + 2 + 1 + 1)/n

	# print(out our table header
	print('\n\nNewton\'s Method Line Search on %s' \
		% fobj.descriptive_name())
	print('alpha_0 = %f, x_0 = %s' % (alpha_0, x_0))
	print('iteration\talpha_k\t\tx_k\t\t\t\tf evals')

	#initialize reporting dict to capture runtime parameters
	rep_dict = {}

	# repeat loop until $\|\nabla f(x_k)\| \leq$ xtol
	while dot(fobj.grad_f(x_k), fobj.grad_f(x_k)) > xtol:
		print('\n begin iteration {}'.format(iterations))
		iter_rep_dict = {}
		iter_rep_dict['x_k'] = x_k
		print(fobj.grad_f(x_k))
		print(fobj.hessian_f(x_k))
		# $p_k = -B_k^{-1}\nabla f_k = -\nabla^2 f(x_k) \nabla f_k$
		p_k = -1.0*np.matmul(linalg.inv(fobj.hessian_f(x_k)), \
							  fobj.grad_f(x_k))
		# apply the line search technique supplied
		alpha_k = f_alpha_k(fobj, x_k, p_k, alpha_0)
		# $x_{k+1} = x_k + \alpha_k p_k$
		x_k = x_k + (alpha_k * p_k)

		#add the sotrage in the current interation for p_k, alpha_k and x_k
		# p_k should also be same dimension as x_k, and alpha_k is a scalar, so 1
		storage = storage + fobj.storage_count()

		iter_rep_dict['alpha_k'] = alpha_k
		iter_rep_dict['f_x_k'] = fobj.f(x_k)
		iter_rep_dict['grad_f_x_k'] = fobj.grad_f(x_k)
		iter_rep_dict['storage'] = storage
		iter_rep_dict['f_eval'] = fobj.eval_count()[0]
		iter_rep_dict['grad_f_eval'] = fobj.eval_count()[1]
		iter_rep_dict['hessian_f_eval'] = fobj.eval_count()[2]
		rep_dict[iterations] = iter_rep_dict

		# print(out the line of the table
		# print('%d\t\t%s\t%s\t%f\t%d\t%d\t%d\t%d' \
        #             % (iterations, x_k, p_k, alpha_k, fobj.eval_count()[0], fobj.eval_count()[1], fobj.eval_count()[2], storage))
		for k, v in iter_rep_dict.items():
			print('{}: {}'.format(k, v))
		print('\n end iteration {} \n************\n**************\n'.format(iterations))

		# make sure that this line gets written to stdout
		sys.stdout.flush()

		iterations += 1
		

		# storing the function, gradient and hessian evalution results at each

	return rep_dict


def SteepestDescentLineSearch(x_0, alpha_0, f_alpha_k, fobj, xtol):
	"""
	Uses the steepest decent line search, in which $B_k = I$,
	to find the minimum value of the objective function.
	
	Inputs:
		x_0 is the Numeric array giving the value of $x_0$
		
		alpha_0 is a floating point scalar giving the value of $\alpha_0$
		
		f_alpha_k is a function reference, a function that takes an
			$x_k$, a $p_k$, and an $\overline{\alpha}$ and returns
			an $alpha_k$.  For reference see BacktrackingAlpha above.


		fobj is an ObjectiveFunction object.  The functions must take 
			inputs of the same degree as x_0.
	
		xtol is the tolerance on the stationary point.  The 
			algorithm will terminate when $\|\nabla f(x_k)\|^2 \leq$ xtol.
	
	Outputs:
		Returns the computed value of the stationary point of $f$, $x^\ast$
		within tolerance xtol.
		
	Side Effects:
		Writes the values of $x_k$, $\alpha_k$, and $p_k$ at each iteration
		to stdout.
	"""
	x_k = x_0
	iterations = 0

	#initialize dimensionality and storage in terms of n (the dimensionality of the problem)
	n = fobj.n
	# so (len(x_k) + 2 floating values for pk + 1 floating value for alpha_k + 1 floating value for iteration count)/n;
	# the division by n is to present thee result in terms of dimensionality of the problem
	storage = (len(x_k) + 2 + 1 + 1)/n

	# print(out our table header
	print('\n\nSteepest Descent Line Search on %s' \
		% fobj.descriptive_name())
	print('alpha_0 = %f, x_0 = %s' % (alpha_0, x_0))
	print('iteration\talpha_k\t\tx_k\t\t\t\tf evals')

	#initialize reporting dict to capture runtime parameters
	rep_dict = {}

	# repeat loop until $\|\nabla f(x_k)\| \leq$ xtol
	while dot(fobj.grad_f(x_k), fobj.grad_f(x_k)) > xtol:
		print('\n begin iteration {}'.format(iterations))
		iter_rep_dict = {}
		iter_rep_dict['x_k'] = x_k

		# $p_k = -B_k^{-1}\nabla f_k = -\nabla f_k$
		p_k = -1.0 * fobj.grad_f(x_k)
		# apply the line search technique supplied
		alpha_k = f_alpha_k(fobj, x_k, p_k, alpha_0)
		# $x_{k+1} = x_k + \alpha_k p_k$
		x_k = x_k + (alpha_k * p_k)
		
		#add the sotrage in the current interation for p_k, alpha_k and x_k
		# p_k should also be same dimension as x_k, and alpha_k is a scalar, so 1
		storage = storage + fobj.storage_count()

		iter_rep_dict['alpha_k'] = alpha_k
		iter_rep_dict['f_x_k'] = fobj.f(x_k)
		iter_rep_dict['grad_f_x_k'] = fobj.grad_f(x_k)
		iter_rep_dict['storage'] = storage
		iter_rep_dict['f_eval'] = fobj.eval_count()[0]
		iter_rep_dict['grad_f_eval'] = fobj.eval_count()[1]
		iter_rep_dict['hessian_f_eval'] = fobj.eval_count()[2]
		rep_dict[iterations] = iter_rep_dict

		# print(out the line of the table
		# print('%d\t\t%s\t%s\t%f\t%d\t%d\t%d\t%d'
        #             % (iterations, x_k, p_k, alpha_k, fobj.eval_count()[0], fobj.eval_count()[1], fobj.eval_count()[2], storage))
		# make sure that this line gets written to stdout
		#
		for k, v in iter_rep_dict.items():
			print('{}: {}'.format(k, v))
		print('\n end iteration {} \n************\n**************\n'.format(iterations))

		iterations += 1
		sys.stdout.flush()
		
	return rep_dict

def zoom(fobj, x_k, p_k, alpha_lo, alpha_hi, phi_0, phi_prime_0, c_1, c_2):
	"""
	This is an implementation of Algorithm 3.3, page 60 of Nocedal and Wrigt
	which is a subprocedure to implement the Strong Wolfe Conditions line 
	search provided below in the StrongWolfe function.  
	
	Inputs:
		fobj is an objective function object
		
		x_k is the $x_k$ value, a Numeric array, used to find $\alpha$
		
		p_k is the $p_k$ value, a Numeric array, used to find $\alpha$
		
		alpha_lo is the $\alpha_{lo}$ value, a floating point scalar
		
		alpha_hi is the $\alpha_{hi}$ value, a floating point scalar
		
		phi_0 is $\phi(0)$, a floating point scalar
		
		phi_prime_0 is $\phi'(0)$, a floating point scalar
		
		c_1 is the constant $c_1$ from the Strong Wolfe conditions
		
		c_2 is the constant $c_2$ from the Strong Wolfe conditions
	
	Returns:
		A suitable value for $\alpha$ that meets the Strong Wolfe 
			conditions, a floating point scalar.
			
	Side Effects:
		None
	"""	
	
	while 1:
		# interpolate using bisection to find a trial step 
		# length $\alpha_j$ between $\alpha_{lo}$ and $\alpha_{hi}$
		alpha_j = (alpha_hi + alpha_lo) / 2.0
			
		# on the rare occasion that $\alpha_{hi} = \alpha_{lo}$ we actually go outside that range to find a usable value.
		if alpha_hi == alpha_lo:
			alpha_j = alpha_hi / 2.0
			
		# Evaluate $\phi(\alpha_j)$
		phi_alpha_j = fobj.f(x_k + alpha_j*p_k)
		
		# if $\phi(\alpha_j) > \phi(0) + c_1\alpha_j\phi'(0)$ or $\phi(\alpha_j) \geq \phi(\alpha_{lo})$
		if phi_alpha_j > phi_0 + c_1*alpha_j*phi_prime_0 or phi_alpha_j >= fobj.f(x_k + alpha_lo*p_k):
			# $\alpha_{hi} \leftarrow \alpha_j$
			alpha_hi = alpha_j
		else:
			# Evaluate $\phi'(\alpha_j)$
			phi_prime_alpha_j = dot(fobj.grad_f(x_k + alpha_j*p_k), p_k)
			
			# if $|\phi'(\alpha_j)| \leq -c_2 \phi'(0)$
			if abs(phi_prime_alpha_j) <= -c_2*phi_prime_0:
				# set $\alpha_\ast \leftarrow \alpha_j$ and stop
				return alpha_j
			
			# if $\phi'(\alpha_j)(\alpha_{hi} - \alpha_{lo}) \geq 0$
			if phi_prime_alpha_j*(alpha_hi - alpha_lo) >= 0:
				# $\alpha_{hi} \leftarrow \alpha_{lo}$
				alpha_hi = alpha_lo
			
			# $\alpha_{lo} \leftarrow \alpha_j$
			alpha_lo = alpha_j

def StrongWolfe(fobj, x_k, p_k, alpha_max):
	"""
	This is an implementation of Algorithm 3.2, page 59 of Nocedal and Wright,
	a line search algorithm that meets the Strong Wolfe conditions.
	
	Inputs:
		fobj is an objective function object
		
		x_k is the $x_k$ value, the staring point to be used for the 
			search for $\alpha$, and is a Numeric array
			
		p_k is the $p_k$ value, the search direction to be used for the 
			search for $\alpha$, and is a Numeric array
			
		alpha_max is some maximum value that the algorithm will not exceed.

	Returns:
		A suitable value for $\alpha$, a floating point scalar, that 
			satisfiies the Strong Wolfe Conditions
	
	Side Effects:
		None
	"""
	# set $\alpha_0 \leftarrow 0$, choose $\alpha_1 > 0$ and $\alpha_{max}$
	alpha_i_minus_1 = 0
	alpha_i = alpha_max
	
	# $i \leftarrow 1$
	i = 1

	# pre-eval $\phi(0)$
	phi_0 = fobj.f(x_k)

	# from (A.16) we have $\phi'(\alpha) = \nabla f(x_k + \alpha p_k)^T p_k$
	# so $\phi'(0) = \nabla f_k^Tp_k$
	phi_prime_0 = dot(fobj.grad_f(x_k), p_k)

	phi_alpha_i_minus_1 = phi_0

	# intelligent values for these found on pg 37-38
	c_1 = 10e-4
	c_2 = 0.99

	while 1:
		# Evaluate $\phi(\alpha_i)$
		phi_alpha_i = fobj.f(x_k + alpha_i * p_k)
		
		# if $\phi(\alpha_i) > \phi(0) + c_1\alpha_i\phi'(0)$ or $[\phi(\alpha_i) \geq \phi(\alpha_{i-1})$ and $i > 1]$
		if phi_alpha_i > phi_0 + c_1*alpha_i*phi_prime_0 or (phi_alpha_i >= phi_alpha_i_minus_1 and i > 1):
			# $\alpha_\ast \leftarrow$ zoom$(\alpha_{i-1}, \alpha_i)$ and stop;
			return zoom(fobj, x_k, p_k, alpha_i_minus_1, alpha_i, phi_0, phi_prime_0, c_1, c_2)
		
		# Evaluate $\phi'(\alpha_i) = \nabla f(x_k + \alpha_i p_k)^Tp_k$
		phi_prime_alpha_i = dot(fobj.grad_f(x_k + alpha_i*p_k), p_k)
		
		# if $|\phi'(a_i)| \leq -c_2\phi'(0)$
		if abs(phi_prime_alpha_i) <= -c_2*phi_prime_0:
			# set $\alpha_\ast \leftarrow \alpha_i$ and stop;
			return alpha_i	
		
		# if $\phi'(\alpha_i) \geq 0$
			# set $\alpha_\ast \leftarrow$ zoom$(\alpha_i, \alpha_{i-1})$ and stop
			return zoom(fobj, x_k, p_k, alpha_i, alpha_i_minus_1, phi_0, phi_prime_0, c_1, c_2)
		
		# advance our variables
		alpha_i_minus_1 = alpha_i
		phi_alpha_i_minus_1 = phi_alpha_i
		
		# Choose $\alpha_{i+1} \in (\alpha_i, \alpha_{max})$
		alpha_i = min(alpha_max, 1.1*alpha_i)
		
		# $i \leftarrow i + 1$
		i += 1
	

def BFGSLineSearch(x_0, alpha_0, f_alpha_k, fobj, epsilon, H_0):
	"""
	An implementation of the modified BFGS algorithm
	from Nocedal and Wright page 198.  
	
	Inputs:
		x_0 is a Numeric array giving the initial value $x_0$

		alpha_0 is the $\alpha_0$, the inital value for $\alpha$.

		f_alpha_k is the line search function, it takes a fobj,
			x_k, p_k, and alpha_max and returns an alpha value
			for the current iteration.  Recommend StrongWolfe above.

		fobj is the objective function object

		epsilon is the tolerance value of $\epsilon$, the function
			returns when $\|\nabla f_k\| > \epsilon$
		
		H_0 is the inital inverse Hessian estimate $H_0$
	
	Outputs:
		Returns the computed value of the stationary point of $f$, $x^\ast$,
			within tolerance $\epsilon$

	Side Effects:
		Reports to stdout at each iteration the value of $\alpha_k$,
			$x_k$, the number of function evaluations, and 
			if the product $y_k^Ts_k$ is positive.	
	"""
	# $k \leftarrow 0$
	k = 0

	x_k = x_0
	H_k = H_0

	#initialize dimensionality and storage in terms of n (the dimensionality of the problem)
	n = fobj.n
	# so (len(x_k) + 2 floating values for pk + 1 floating value for alpha_k + 1 floating value for iteration count)/n;
	# the division by n is to present thee result in terms of dimensionality of the problem
	storage = (len(x_k) + 2 + 1 + 1 + n*len(H_k))/n

	# print(out our table header
	print('\n\nBFGS Line Search on %s' \
		% fobj.descriptive_name())
	print('alpha_0 = %f, x_0 = %s' % (alpha_0, x_0))
	#print('iterations\talpha_k\t\tx_k\t\t\t\tf_evals\tyk*sk positive?')

	#initialize reporting dict to capture runtime parameters
	rep_dict = {}

	# while $\|\nabla f_k\|$
	while dot(fobj.grad_f(x_k), fobj.grad_f(x_k)) > epsilon:
		print('\n begin iteration {}'.format(k))
		iter_rep_dict = {}
		iter_rep_dict['x_k'] = x_k
		# $p_k = -H_k \nabla f_k$
		p_k = -1.0 * np.matmul(H_k, fobj.grad_f(x_k))
		# apply the line search technique supplied
		alpha_k = f_alpha_k(fobj, x_k, p_k, alpha_0)
		# $x_{k+1} = x_k + \alpha p_k$
		x_k_plus_1 = x_k + (alpha_k * p_k)
		# $s_k = x_{k+1} - x_k$
		s_k = x_k_plus_1 - x_k
		# $y_k = \nabla f_{k+1} - \nabla f_k$
		y_k = fobj.grad_f(x_k_plus_1) - fobj.grad_f(x_k)
		# $\rho_k = (y_k^T s_k)^{-1}$
		rho_k = 1.0 / dot(y_k, s_k)
		# $H_{k+1} = (I - \rho_k s_k y_k^T)H_k(I - \rho_ky_ks_k^T) + p_ks_ks_k^T$
		
		# H_k_plus_1 = np.matmul(\
		# 	np.matmul(identity(x_0.shape[0]) - \
		# 		rho_k*np.matmul(\
		# 		s_k, transpose(y_k)), H_k),\
		# 	identity(x_0.shape[0]) \
		# 	 - rho_k*np.matmul(y_k, transpose(s_k))) \
		# 	+ np.matmul(np.matmul(p_k, transpose(s_k)), \
		# 			s_k)
		H_k_plus_1 = np.matmul(\
			np.matmul(identity(x_0.shape[0]) - \
				rho_k*np.matmul(\
				s_k, transpose(y_k)), H_k),\
			identity(x_0.shape[0]) \
			 - rho_k*np.matmul(y_k, transpose(s_k))) \
			+ rho_k*np.matmul(s_k, transpose(s_k))

		# $k \leftarrow k+1$
		#c.interact(local=locals(), banner='inspecting bfgs')
		k = k + 1
		# advance the values
		x_k = x_k_plus_1
		H_k = H_k_plus_1		

		# check and make sure that $y_k^Ts_k$ is positive
		if dot(y_k, s_k) > 0:
			y_k_s_k_positive = 'yes'
		else:
			y_k_s_k_positive = 'no'
		
		#add the sotrage in the current interation for p_k, alpha_k and x_k
		# p_k should also be same dimension as x_k, and alpha_k is a scalar, so 1
		storage = storage + fobj.storage_count()

		iter_rep_dict['alpha_k'] = alpha_k
		iter_rep_dict['f_x_k'] = fobj.f(x_k)
		iter_rep_dict['grad_f_x_k'] = fobj.grad_f(x_k)
		iter_rep_dict['storage'] = storage
		iter_rep_dict['f_eval'] = fobj.eval_count()[0]
		iter_rep_dict['grad_f_eval'] = fobj.eval_count()[1]
		iter_rep_dict['hessian_f_eval'] = fobj.eval_count()[2]
		rep_dict[k-1] = iter_rep_dict

		for ky, v in iter_rep_dict.items():
			print('{}: {}'.format(ky, v))
		print('\n end iteration {} \n************\n**************\n'.format(k))

		# print(out the line of the table
		# print('%d\t\t%s\t%s\t%f\t%d\t%d\t%d\t%d'
        #             % (k, x_k, p_k, alpha_k, fobj.eval_count()[0], fobj.eval_count()[1], fobj.eval_count()[2], storage, y_k_s_k_positive))
		
		sys.stdout.flush()
	
	return rep_dict

#####################
# unit testing code #
#####################

import unittest
import ObjectiveFunctions

class BFGSTestCase(unittest.TestCase):
	def testDescentFunction(self):
		x_0 = array((stats.rand.uniform(-100.0, 100.0)[0], \
			stats.rand.uniform(-100.0, 100.0)[0]))
		alpha_0 = 1.0
		f1 = ObjectiveFunctions.testObjFunc()
		
		
		x_star = BFGSLineSearch(x_0, alpha_0, \
					StrongWolfe, 
					f1, \
					9.9e-13, \
					linalg.inv(f1.hessian_f(x_0)))
				
		assert x_star == array((0.0, 0.0))
		
		x_0 = array((stats.rand.uniform(-100.0, 100.0)[0], \
			stats.rand.uniform(-100.0, 100.0)[0]))
		alpha_0 = 1.0
		f2 = ObjectiveFunctions.testSecondFunc()
			
		x_star = BFGSLineSearch(x_0, alpha_0, \
					StrongWolfe, \
					f2, \
					9.9e-13, \
					linalg.inv(f2.hessian_f(x_0)))
					
		assert alltrue(x_star == array((1.0, 1.0)))

class SteepestDescentTestCase(unittest.TestCase):
	def testDescentFunction(self):
		x_0 = array((stats.rand.uniform(-100.0, 100.0)[0], \
			stats.rand.uniform(-100.0, 100.0)[0]))
		alpha_0 = 1.0
		
		x_star = SteepestDescentLineSearch(x_0, alpha_0, \
				  BacktrackingAlpha, \
				  ObjectiveFunctions.testObjFunc(), \
				  9.9e-13)
		
		assert alltrue(x_star == array((0.0, 0.0)))

class NewtonsMethodTestCase(unittest.TestCase):
	def testDescentFunction(self):
		x_0 = array((stats.rand.uniform(-100.0, 100.0)[0], \
			stats.rand.uniform(-100.0, 100.0)[0]))
		alpha_0 = 1.0
		
		x_star = NewtonsMethodLineSearch(x_0, alpha_0, \
				BacktrackingAlpha, \
				ObjectiveFunctions.testObjFunc(), \
				9.9e-13)
				
		assert alltrue(x_star == array((0.0, 0.0)))
		
		x_0 = array((stats.rand.uniform(-100.0, 100.0)[0], \
			stats.rand.uniform(-100.0, 100.0)[0]))
		alpha_0 = 1.0	
		
		x_star = NewtonsMethodLineSearch(x_0, alpha_0, \
				 BacktrackingAlpha, \
				 ObjectiveFunctions.testSecondFunc(), \
				 9.9e-13)
		
		assert alltrue(x_star == array((1.0, 1.0)))

if __name__ == '__main__':
	unittest.main()
