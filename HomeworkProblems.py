from LineSearch import *
from TrustRegion import *
from ObjectiveFunctions import *
from ConjugateGradient import *
from scipy import *

def problem_3_1():
	x_0 = array((1.2, 1.2))
	alpha_0 = 1

	# does not converge quickly, even after 5500 iterations
	# it only has $x^\ast$ calculated to 4 decimal places
	x_star = SteepestDescentLineSearch(x_0, alpha_0, \
			BacktrackingAlpha, \
			Rosenbrock(), \
			9.9e-4)

	x_0 = array((1.2, 1.2))
	alpha_0 = 1

	x_star = NewtonsMethodLineSearch(x_0, alpha_0,\
			   BacktrackingAlpha, \
			   Rosenbrock(), \
			   9.9e-13)

	x_0 = array((-1.2, 1))
	alpha_0 = 1

	# again, doesn't converge in a tractable amount of time
	x_star = SteepestDescentLineSearch(x_0, alpha_0, \
			BacktrackingAlpha,\
			Rosenbrock(), \
			9.9e-4)

	x_0 = array((-1.2, 1))
	alpha_0 = 1

	x_star = NewtonsMethodLineSearch(x_0, alpha_0,\
			   BacktrackingAlpha, \
			   Rosenbrock(), \
			   9.9e-13)

def problem_3_9():
	x_0 = array((1.2, 1.2))
	alpha_0 = 1
	r = Rosenbrock()

	x_star = BFGSLineSearch(x_0, alpha_0, \
			StrongWolfe, \
			r, \
			9.9e-13, \
			linalg.inv(r.hessian_f(x_0)))

	x_0 = array((-1.2, 1))

	x_star = BFGSLineSearch(x_0, alpha_0, \
				StrongWolfe,
				r,
				9.9e-13, \
				linalg.inv(r.hessian_f(x_0)))

def problem_4_1():
	# we need the solution of $p_k$ as a function of $\Delta$
	print 'x = (0, -1)'
	print 'Delta\tp_k'
	p = Problem4_1(2)
	B_k = p.hessian_f(array((0, -1)))
	epsilon = 9.9e-13
	g = p.grad_f(array((0, -1)))
	for Delta in arange(0, 2, 0.01):
		p_k, t, b, n, m = CG_Steihaug(p, epsilon, Delta, B_k, g)
		print '%f\t%s' % (Delta, p_k)
	
	print 'x = (0, 0.5)'
	print 'Delta\tp_k'
	p = Problem4_1(2)
	B_k = p.hessian_f(array((0, 0.5)))
	epsilon = 9.9e-13
	g = p.grad_f(array((0, 0.5)))
	for Delta in arange(0, 2, 0.01):
		p_k, t, b, n, m = CG_Steihaug(p, epsilon, Delta, B_k, g)
		print '%f\t%s' % (Delta, p_k)

	

def problem_4_3():
	maxiter = 10000

	print '***** Trust Region ***** for n = 10:'
	for i in range(0, 1):
		x_0 = []
		for j in range(0, 10):
			x_0.append(stats.rand.uniform(0, 2)[0])
		x_0 = transpose(array((x_0)))
		
		print 'x_0: %s' % x_0

		x_star = TrustRegion(x_0, 10.0, 1.0, 0.1, Problem4_3(10), 1000)
		
		print 'x_star: %s' % x_star

	print '***** Trust Region ***** for n = 50:'
	for i in range(0, 1):
		x_0 = []
		for j in range(0, 50):
			x_0.append(stats.rand.uniform(0, 2)[0])
		x_0 = transpose(array((x_0)))

		print 'x_0: %s' % x_0

		x_star = TrustRegion(x_0, 10.0, 1.0, 0.1, Problem4_3(50), 1000)

		print 'x_star: %s' % x_star


def problem_5_1():
	print 'n\titerations'
	print '5\t%s' % GetCGIterations(5)
	print '8\t%s' % GetCGIterations(8)
	print '12\t%s' % GetCGIterations(12)
	print '20\t%s' % GetCGIterations(20)
	print '100\t%s' % GetCGIterations(100)


if __name__ == '__main__':
#	problem_3_1()
#	problem_3_9()
	problem_4_1()
#	problem_4_3()
#	problem_5_1()
