from scipy import *

class ObjectiveFunction:
	"""
	This is the base class for the objective functions that we
	will evaluate for optimization.  This class ensures that a function, 
	its exact gradient and its exact Hessian are grouped into a single, 
	logical unit.
	
	Furthermore, this base class tracks the number of unique function 
	evaluations required on a given optimization run.  
	
	It does this by storing the supplied inputs $x$ to each function
	in an array.  If the input $x$ is already in the array, it is
	not added again.  Therefore, the number of unique function
	evaluations is the size of the array.	
	"""
	def __init__(self, ninit = 0):
		self.f_cache = []
		self.f_grad_cache = []
		self.f_hessian_cache = []
		self.f_val_cache = {}
		self.f_grad_val_cache = {}
		self.f_hessian_val_cache = {}
		self.n = ninit

	def __check_n(self, x):
		"""
		The first time an x is supplied to either real_f, grad_f,
		or hessian_f, we extract the dimension of this x and set
		our internal n value to it.  
		
		Or, for functions of fixed dimesions, they may set n
		in their constructor.
		
		
		This function either sets the n value if it hasn't been set, 
		or compares the dimension of the x supplied to our n 
		value to make sure that an x of appropriate dimension 
		has been supplied.
		"""
		if self.n:
			if x.shape[0] != self.n:
				raise 'bad dimension on x'
		else:
			self.n = x.shape[0]


	def descriptive_name(self):
		"""
		Return a descriptive name of the objective function, for
		use in the table headers.
		"""
		return 'generic objective function'

	def eval_count(self):
		"""
		Returns an array with the number of times f, grad_f, and hessian_f have
		been called with unique inputs.
		"""
		return [len(self.f_cache), len(self.f_grad_cache), len(self.f_hessian_cache)]
	
	def storage_count(self):
		"""
		Returns an array with the storage of f, grad_f, and hessian_f for each 
		x_k encountered.
		"""
		return len(self.f_val_cache) + len(self.f_grad_cache) + self.n*len(self.f_hessian_val_cache)


	def real_f(self, x):
		"""
		The actual function you want to override in a subclass.

		Should return the value of the function evaluated at $x$.
		"""
		return 0

	def f(self, x):
		"""
		Returns the value of the function evaluated at $x$.

		Uses caching to count the number of unique function
		evaluations.

		This is the function that the optimization methods
		should call.
		"""
		self.__check_n(x)
		
		for cache_x in self.f_cache:
			# x == cache_x on array types returns a matrix
			# of value-by-value comparisons.  The alltrue
			# numpy function returns true iff all comparisons
			# returned true
			if alltrue(cache_x == x):
				#return self.real_f(x)
				return self.f_val_cache[tuple(x)]
		
		self.f_cache.append(x)
		self.f_val_cache[tuple(x)] = self.real_f(x)
		
		return self.real_f(x)

	def real_grad_f(self, x):
		"""
		The actual functino you want to override in a subclass.

		Should return the value of the exact gradient of the
		function $f$ evaluated at $x$.
		"""
		return 0

	def grad_f(self, x):
		"""
		Returns the value of the gradient of the function f
		evaluated at $x$.
		"""
		self.__check_n(x)
		
		for cache_x in self.f_grad_cache:
			if alltrue(cache_x == x):
				#return self.real_grad_f(x)
				return self.f_grad_val_cache[tuple(x)]

		self.f_grad_cache.append(x)
		self.f_grad_val_cache[tuple(x)] = self.real_grad_f(x)

		return self.real_grad_f(x)

	def real_hessian_f(self, x):
		"""
		The actual function you want to override in a subclass.

		Returns the value of the exact Hessian of the function
		$f$ evaluated at $x$.
		"""
		return 0
	
	def hessian_f(self, x):
		"""
		Returns the value of the Hessian of the function f
		evaluated at $x$.
		"""
		self.__check_n(x)

		for cache_x in self.f_hessian_cache:
			if alltrue(cache_x == x):
				#return self.real_hessian_f(x)
				return self.f_hessian_val_cache[tuple(x)]
		
		self.f_hessian_cache.append(x)
		self.f_hessian_val_cache[tuple(x)] = self.real_hessian_f(x)

		return self.real_hessian_f(x)

class Rosenbrock(ObjectiveFunction):
	"""
	The Rosenbrock function is defined in Nocedal and Wright's
	textbook Numerical Optimization, Formula 2.23, on page 30.

	The formula has been rewritten slightly to account for the fact
	that the input array has two elements at indicies 0 and 1, whereas
	the formula defines two elements $x_1$ and $x_2$.  The reader should
	consider code $x_0$ = book $x_1$ and code $x_1$ = book $x_2$. 
	Therefore we write it as:

	$f(x) = 100(x_1 - x_0^2)^2 + (1 - x_0)^2$	
	"""
	def __init__(self):
		ObjectiveFunction.__init__(self)
		self.n = 2
		
	def descriptive_name(self):
		"""
		Return a descriptive name of the objective function, for
		use in the table headers.
		"""
		return 'the Rosenbrock function'

	def real_f(self, x):
		"""
		Returns the value of Rosenbrock function evaluated at $x$.
	
		Inputs:
			$x$ is a 2D Numeric array of floating point numbers which 
			represent the inputs to the Rosenbrock function.

		Outputs:
			Returns a floating point scalar with the value of the 
			Rosenbrock function evaluated at $x$.    

		Side Effects:
			None.
		"""
		return (100 * (x[1] - x[0]**2)**2) + (1 - x[0])**2

	def real_grad_f(self, x):
		"""
		Returns the value of the gradient of the Rosenbrock function 
		evaluated at $x$.  

		The gradient of the Rosenbrock function was calculated by hand
		to be:
		
		$\nabla f(x) = \left[ \begin{array}{c}
		400x_0^3 - 400x_1x_0 + 2x_0 - 2 \\
		-200x_0^2 + 200x_1 \end{array} \right]$

		Inputs:
            		$x$ is a 2D Numeric array of floating point numbers 
				which represent the inputs to the Rosenbrock 
				gradient function
			
		Outputs:
			Returns a 2D Numeric array with the values of the 
			gradient of the Rosenbrock function evaluated at $x$.

		Side Effects:
			None.
	    """
		return array((400*x[0]**3 - 400*x[1]*x[0] + 2*x[0] - 2,
					  -200*x[0]**2 + 200*x[1]))

	def real_hessian_f(self, x):
		"""
		Returns the value of the Hessian of the Rosenbrock function
		evaluated at $x$.

		The Hessian of the Rosenbrock function was calculated by hand
		to be:

		$\nabla^2 f(x) = \left[ \begin{array}{cc}
		1200x_0^2 - 400x_1 + 2  & -400x_0 \\
		-400x_0 & 200 \end{array} \right]$

		Inputs:
			$x$ is a 2D Numeric array of the floating point numbers
			which represent the inputs to the Rosenbrock Hessian
			function.

		Outputs:
			Returns a 2x2 Numeric matrix with the values of
			the Hessian of the Rosenbrock function evaluated at $x$.

		Side Effects:
			None.
		"""
		return array(([1200*x[0]**2 - 400*x[1] + 2, -400*x[0]],
					  [-400*x[0], 200]))

class Problem4_1(ObjectiveFunction):
	"""
	This class is an implementation of the function defined in 
	Problem 4.1 of Nocedal and Wright, on page 97.
	"""
	def descriptive_name(self):
		return 'the function defined in problem 4.1'

	def real_f(self, x):
		"""
		Returns the value of the function evaluated at $x$.
		
		The function is defined to be:
		
		$f(x) = 10(x_2 - x_1^2)^2 + (1 - x_1)^2$
       		"""
		return 10*(x[1] - x[0]**2)**2 + (1 - x[0])**2

	def grad_f(self, x):
		"""
		Returns the value of the gradient of the function,
		$\nabla f$ evaluated at $x$.
		
		The gradient is defined to be:
		
		$\nabla f(x) = \left[ \begin{array}{c}
		  -40x_1x_2 + 40x_1 - 2 + 2x_1 \\ 20x_2 - 20x_1^2 \end{array} \right]$
		"""
		return array((-40.0*x[0]*x[1] + 40*x[0] - 2 + 2*x[0],
			      20*x[1] - 20*x[0]))

	def hessian_f(self, x):
		"""
		Returns the value of the Hessian of the function,
		$\nabla^2 f$ evaluated at $x$.

		The Hessian is defined to be:

		$\nabla^2 f(x) = \left[ \begin{array}{cc}
		  -40x_2 + 42 & -40x_1 \\ -40x_1 & 20 \end{array} \right]$
		"""
		return array(([-40*x[1]+42, -40*x[0]], [-40*x[0], 20]))


class Problem4_3(ObjectiveFunction):
	"""
	This class is an implementation of the function defined in 
	Problem 4.3 of Nocedal and Wright, on page 97.
	"""	
				
	def descriptive_name(self):
		return 'the function defined in problem 4.3, n = %d' % self.n

	def real_f(self, x):
		"""
		Returns the value of the function evaluated at x.  Note that the
		dimension on x is taken to be twice the value of the $n$ in the
		summation, as that is the smallest size vector that could be
		processed in this summation.  
		
		The function is defined to be:
		
		$f(x) = \sum_{i=1}^n \left[(1-x_{2i-1})^2 + 10(x_{2i} - x_{2i-1}^2)^2 \right]$
		"""
		accum = 0.0
		
		for i in range(0, self.n/2):
			accum += (1 - x[2*i])**2 + 10*(x[2*i+1] - x[2*i]**2)**2
		
		return accum
		
	def real_grad_f(self, x):
		"""
		Returns the value of the gradient of the function evaluated at x.
		The $i$th component of the gradient, for $i$ valid from $0$ to
		$2n -1$ was calculated by hand to be:
		
		$\nabla f_i(x) = \left\{ \begin{array}{rcl}
			-2 + 2x_i - 40x_ix_{i+1} + 40x_i^3 & \mbox{for} & i \mbox{ even} \\
			20x_i - 20x_{i-1}^2                & \mbox{for} & i \mbox{ odd} \end{array} \right.$
		"""
		grad_transpose = []
		
		for i in range(0, self.n):
			if not i % 2:
				# i even
				grad_transpose.append(-2 + 2*x[i] \
				  - 40*x[i+1]*x[i] + 40*x[i]**3)
			else:
				# i odd
				grad_transpose.append(20*x[i] - 20*x[i-1]**2)
		
		return transpose(array((grad_transpose)))
		
	def real_hessian_f(self, x):
		"""
		Returns the hessian of the function evaluated at $x$.

		Just constructs the diagonal terms, then uses numpy's
		diag() operator to construct an nxn matrix with hessian
		as the diagonal members of that matrix.
		
		The $ij$ compmonent of the hessian was calculated by 
		hand to be:
		
		$\nabla^2 f_{i,j}(x) = \left\{ \begin{array}{rcll}
			2 - 40x_{i+1} + 120x_i^2 & \mbox{for} & i=j & i \mbox{ even} \\
			20 & \mbox{for} & i=j & i \mbox{ odd} \\
			0 & & & \mbox{otherwise} \end{array} \right.$
		"""
		hessian = []

		for i in range(0, self.n):
			if not i % 2:
				hessian.append(2.0 - 40*x[i+1] + 120*x[i]**2)
			else:
				hessian.append(20.0)
		
		return diag(array((hessian)))


class SphereFunction(ObjectiveFunction):
	"""
	The Sphere function is defined on wikipedia and can be found at
	https://en.wikipedia.org/wiki/Test_functions_for_optimization

	The formula has been rewritten slightly to account for the fact
	that the input array has two elements at indicies 0 and 1, whereas
	the formula defines two elements $x_1$ and $x_2$.  The reader should
	consider code $x_0$ = wiki $x_1$ and code $x_1$ = wiki $x_2$. 
	Therefore we write it as:

	$f(x) = (x_0^2 + x_1^2)$	
	"""

	def __init__(self):
		ObjectiveFunction.__init__(self)
		self.n = 2

	def descriptive_name(self):
		"""
		Return a descriptive name of the objective function, for
		use in the table headers.
		"""
		return 'the Sphere function 2d'

	def real_f(self, x):
		"""
		Returns the value of Sphere function evaluated at $x$.
	
		Inputs:
			$x$ is a 2D Numeric array of floating point numbers which 
			represent the inputs to the Sphere function.

		Outputs:
			Returns a floating point scalar with the value of the 
			Sphere function evaluated at $x$.    

		Side Effects:
			None.
		"""
		return ((x[0]**2 + x[1]**2))

	def real_grad_f(self, x):
		"""
		Returns the value of the gradient of the Sphere function 
		evaluated at $x$.  

		The gradient of the Sphere function was calculated by hand
		to be:
		
		$\nabla f(x) = \left[ \begin{array}{c}
		2x_0 \\
		2x_1 \end{array} \right]$

		Inputs:
            		$x$ is a 2D Numeric array of floating point numbers 
				which represent the inputs to the Sphere
				gradient function
			
		Outputs:
			Returns a 2D Numeric array with the values of the 
			gradient of the Sphere function evaluated at $x$.

		Side Effects:
			None.
	    """
		#return array((400*x[0]**3 - 400*x[1]*x[0] + 2*x[0] - 2, -200*x[0]**2 + 200*x[1]))
		return array((2*x[0], 2*x[1]))

	def real_hessian_f(self, x):
		"""
		Returns the value of the Hessian of the Sphere function
		evaluated at $x$.

		The Hessian of the Sphere function was calculated by hand
		to be:

		$\nabla^2 f(x) = \left[ \begin{array}{cc}
		1200x_0^2 - 400x_1 + 2  & -400x_0 \\
		-400x_0 & 200 \end{array} \right]$

		Inputs:
			$x$ is a 2D Numeric array of the floating point numbers
			which represent the inputs to the Rosenbrock Hessian
			function.

		Outputs:
			Returns a 2x2 Numeric matrix with the values of
			the Hessian of the Sphere function evaluated at $x$.

		Side Effects:
			None.
		"""
		return array(([2, 0], [0, 2]))


class BealesFunction(ObjectiveFunction):
	"""
	The Beales function is defined on wikipedia and can be found at
	https://en.wikipedia.org/wiki/Test_functions_for_optimization

	The formula has been rewritten slightly to account for the fact
	that the input array has two elements at indicies 0 and 1, whereas
	the formula defines two elements $x_1$ and $x_2$.  The reader should
	consider code $x_0$ = wiki $x_1$ and code $x_1$ = wiki $x_2$. 
	Therefore we write it as:

	$f(x) = (1.5 - x_0 + x_0*x_1)^2 + (2.25 - x_0 + x_0*x_1^2)^2 + (2.625 - x_0 + x_0*x_1^3)^2$		
	"""

	def __init__(self):
		ObjectiveFunction.__init__(self)
		self.n = 2

	def descriptive_name(self):
		"""
		Return a descriptive name of the objective function, for
		use in the table headers.
		"""
		return 'the Beales function 2d'

	def real_f(self, x):
		"""
		Returns the value of Beales function evaluated at $x$.
	
		Inputs:
			$x$ is a 2D Numeric array of floating point numbers which 
			represent the inputs to the Beales function.

		Outputs:
			Returns a floating point scalar with the value of the 
			Beales function evaluated at $x$.    

		Side Effects:
			None.
		"""
		return (((1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*(x[1]**2))**2 + (2.625 - x[0] + x[0]*(x[1]**3))**2))

	def real_grad_f(self, x):
		"""
		Returns the value of the gradient of the Beales function 
		evaluated at $x$.  

		The gradient of the Beales function was calculated by hand
		to be:
		
		[2*(y^2 - 1)*(x*y^2 - x + 2.25) + 2*(y^3 - 1)*(x*y^3 - x + 2.625) + 2*(y - 1)*(x*y - x + 1.5), 
                 2*x*(x*y - x + 1.5) + 4*x*y*(x*y^2 - x + 2.25) + 6*x*y^2*(x*y^3 - x + 2.625)]

		Inputs:
            		$x$ is a 2D Numeric array of floating point numbers 
				which represent the inputs to the Beales
				gradient function
			
		Outputs:
			Returns a 2D Numeric array with the values of the 
			gradient of the Beales function evaluated at $x$.

		Side Effects:
			None.
	    """
		#return array((400*x[0]**3 - 400*x[1]*x[0] + 2*x[0] - 2, -200*x[0]**2 + 200*x[1]))
		return array((2*(x[1]**2 - 1)*(x[0]*x[1]**2 - x[0] + 2.25) + 2*(x[1]**3 - 1)*(x[0]*x[1]**3 - x[0] + 2.625) + 2*(x[1] - 1)*(x[0]*x[1] - x[0] + 1.5), 2*x[0]*(x[0]*x[1] - x[0] + 1.5) + 4*x[0]*x[1]*(x[0]*x[1]**2 - x[0] + 2.25) + 6*x[0]*x[1]**2*(x[0]*x[1]**3 - x[0] + 2.625)))

	def real_hessian_f(self, x):
		"""
		Returns the value of the Hessian of the Beales function
		evaluated at $x$.

		The Hessian of the Beales function was calculated by hand
		to be:

		([2*(x_2 - 1)^2 + 2*(x_2^2 - 1)^2 + 2*(x_2^3 - 1)^2, 6*x_2^2*(x_1*x_2^3 - x_1 + 21/8) - 2*x_1 + 2*x_1*x_2 + 2*x_1*(x_2 - 1) + 4*x_2*(x_1*x_2^2 - x_1 + 9/4) + 4*x_1*x_2*(x_2^2 - 1) + 6*x_1*x_2^2*(x_2^3 - 1) + 3], 
        [ 6*x_2^2*(x_1*x_2^3 - x_1 + 21/8) - 2*x_1 + 2*x_1*x_2 + 2*x_1*(x_2 - 1) + 4*x_2*(x_1*x_2^2 - x_1 + 9/4) + 4*x_1*x_2*(x_2^2 - 1) + 6*x_1*x_2^2*(x_2^3 - 1) + 3, 8*x_1^2*x_2^2 + 18*x_1^2*x_2^4 + 4*x_1*(x_1*x_2^2 - x_1 + 9/4) + 2*x_1^2 + 12*x_1*x_2*(x_1*x_2^3 - x_1 + 21/8)])

		Inputs:
			$x$ is a 2D Numeric array of the floating point numbers
			which represent the inputs to the Rosenbrock Hessian
			function.

		Outputs:
			Returns a 2x2 Numeric matrix with the values of
			the Hessian of the Beales function evaluated at $x$.

		Side Effects:
			None.
		"""
		return array(([2*(x[1] - 1)**2 + 2*(x[1]**2 - 1)**2 + 2*(x[1]**3 - 1)**2, 6*x[1]**2*(x[0]*x[1]**3 - x[0] + 2.625) - 2*x[0] + 2*x[0]*x[1] + 2*x[0]*(x[1] - 1) + 4*x[1]*(x[0]*x[1]**2 - x[0] + 2.25) + 4*x[0]*x[1]*(x[1]**2 - 1) + 6*x[0]*x[1]**2*(x[1]**3 - 1) + 3], [6*x[1]**2*(x[0]*x[1]**3 - x[0] + 2.625) - 2*x[0] + 2*x[0]*x[1] + 2*x[0]*(x[1] - 1) + 4*x[1]*(x[0]*x[1]**2 - x[0] + 2.25) + 4*x[0]*x[1]*(x[1]**2 - 1) + 6*x[0]*x[1]**2*(x[1]**3 - 1) + 3, 8*x[0]**2*x[1]**2 + 18*x[0]**2*x[1]**4 + 4*x[0]*(x[0]*x[1]**2 - x[0] + 2.25) + 2*x[0]**2 + 12*x[0]*x[1]*(x[0]*x[1]**3 - x[0] + 2.625)]))


class GoldsteinPriceFunction(ObjectiveFunction):
	"""
	The GoldsteinPrice function is defined on wikipedia and can be found at
	https://en.wikipedia.org/wiki/Test_functions_for_optimization

	The formula has been rewritten slightly to account for the fact
	that the input array has two elements at indicies 0 and 1, whereas
	the formula defines two elements $x_1$ and $x_2$.  The reader should
	consider code $x_0$ = wiki $x_1$ and code $x_1$ = wiki $x_2$. 
	Therefore we write it as:

	$f(x,y) = ((x + y + 1)^2*(3*x^2 + 6*x*y - 14*x + 3*y^2 - 14*y + 19) + 1)*((2*x - 3*y)^2*(12*x^2 - 36*x*y - 32*x + 27*y^2 + 48*y + 18) + 30)$		
	"""

	def __init__(self):
		ObjectiveFunction.__init__(self)
		self.n = 2

	def descriptive_name(self):
		"""
		Return a descriptive name of the objective function, for
		use in the table headers.
		"""
		return 'the GoldsteinPrice function 2d'

	def real_f(self, x):
		"""
		Returns the value of GoldsteinPrice function evaluated at $x$.
	
		Inputs:
			$x$ is a 2D Numeric array of floating point numbers which 
			represent the inputs to the GoldsteinPrice function.

		Outputs:
			Returns a floating point scalar with the value of the 
			GoldsteinPrice function evaluated at $x$.    

		Side Effects:
			None.
		"""
		return (
                    ((x[0] + x[1] + 1)**2*(3*x[0]**2 + 6*x[0]*x[1] - 14*x[0] + 3*x[1]**2 - 14*x[1] + 19) + 1)*(
                    	(2*x[0] - 3*x[1])**2*(12*x[0]**2 - 36*x[0]*x[1] - 32*x[0] + 27*x[1]**2 + 48*x[1] + 18) + 30)
                )

	def real_grad_f(self, x):
		"""
		Returns the value of the gradient of the GoldsteinPrice function 
		evaluated at $x$.  

		The gradient of the GoldsteinPrice function was calculated by hand
		to be:
		
		too big, calcualte dusing matlab

		Inputs:
            		$x$ is a 2D Numeric array of floating point numbers 
				which represent the inputs to the GoldsteinPrice
				gradient function
			
		Outputs:
			Returns a 2D Numeric array with the values of the 
			gradient of the GoldsteinPrice function evaluated at $x$.

		Side Effects:
			None.
	    """
		#return array((400*x[0]**3 - 400*x[1]*x[0] + 2*x[0] - 2, -200*x[0]**2 + 200*x[1]))
		return array(
                    (((6*x[0] + 6*x[1] - 14)*(x[0] + x[1] + 1)**2 + (2*x[0] + 2*x[1] + 2)*(3*x[0]**2 + 6*x[0]*x[1] - 14*x[0] + 3*x[1]**2 - 14*x[1] + 19))*((2*x[0] - 3*x[1])**2*(12*x[0]**2 - 36*x[0]*x[1] - 32*x[0] + 27*x[1]**2 + 48*x[1] + 18) + 30) + ((x[0] + x[1] + 1)**2*(3*x[0]**2 + 6*x[0]*x[1] - 14*x[0] + 3*x[1]**2 - 14*x[1] + 19) + 1)*((8*x[0] - 12*x[1])*(12*x[0]**2 - 36*x[0]*x[1] - 32*x[0] + 27*x[1]**2 + 48*x[1] + 18) - (2*x[0] - 3*x[1])**2*(36*x[1] - 24*x[0] + 32)), 
					((6*x[0] + 6*x[1] - 14)*(x[0] + x[1] + 1)**2 + (2*x[0] + 2*x[1] + 2)*(3*x[0]**2 + 6*x[0]*x[1] - 14*x[0] + 3*x[1]**2 - 14*x[1] + 19))*((2*x[0] - 3*x[1])**2*(12*x[0]**2 - 36*x[0]*x[1] - 32*x[0] + 27*x[1]**2 + 48*x[1] + 18) + 30) - ((x[0] + x[1] + 1)**2*(3*x[0]**2 + 6*x[0]*x[1] - 14*x[0] + 3*x[1]**2 - 14*x[1] + 19) + 1)*((12*x[0] - 18*x[1])*(12*x[0]**2 - 36*x[0]*x[1] - 32*x[0] + 27*x[1]**2 + 48*x[1] + 18) - (2*x[0] - 3*x[1])**2*(54*x[1] - 36*x[0] + 48)))
                )

	def real_hessian_f(self, x):
		"""
		Returns the value of the Hessian of the GoldsteinPrice function
		evaluated at $x$.

		The Hessian of the GoldsteinPrice function was calculated by hand
		to be:

        way too big, calcualted using matlab

		Inputs:
			$x$ is a 2D Numeric array of the floating point numbers
			which represent the inputs to the Rosenbrock Hessian
			function.

		Outputs:
			Returns a 2x2 Numeric matrix with the values of
			the Hessian of the GoldsteinPrice function evaluated at $x$.

		Side Effects:
			None.
		"""
		return array(
                    ([2*((8*x[0] - 12*x[1])*(12*x[0]**2 - 36*x[0]*x[1] - 32*x[0] + 27*x[1]**2 + 48*x[1] + 18) - (2*x[0] - 3*x[1])**2*(36*x[1] - 24*x[0] + 32))*((6*x[0] + 6*x[1] - 14)*(x[0] + x[1] + 1)**2 + (2*x[0] + 2*x[1] + 2)*(3*x[0]**2 + 6*x[0]*x[1] - 14*x[0] + 3*x[1]**2 - 14*x[1] + 19)) + ((2*x[0] - 3*x[1])**2*(12*x[0]**2 - 36*x[0]*x[1] - 32*x[0] + 27*x[1]**2 + 48*x[1] + 18) + 30)*(12*x[0]*x[1] - 28*x[1] - 28*x[0] + 2*(2*x[0] + 2*x[1] + 2)*(6*x[0] + 6*x[1] - 14) + 6*(x[0] + x[1] + 1)**2 + 6*x[0]**2 + 6*x[1]**2 + 38) + ((x[0] + x[1] + 1)**2*(3*x[0]**2 + 6*x[0]*x[1] - 14*x[0] + 3*x[1]**2 - 14*x[1] + 19) + 1)*(384*x[1] - 256*x[0] - 288*x[0]*x[1] + 24*(2*x[0] - 3*x[1])**2 - 2*(8*x[0] - 12*x[1])*(36*x[1] - 24*x[0] + 32) + 96*x[0]**2 + 216*x[1]**2 + 144), ((8*x[0] - 12*x[1])*(12*x[0]**2 - 36*x[0]*x[1] - 32*x[0] + 27*x[1]**2 + 48*x[1] + 18) - (2*x[0] - 3*x[1])**2*(36*x[1] - 24*x[0] + 32))*((6*x[0] + 6*x[1] - 14)*(x[0] + x[1] + 1)**2 + (2*x[0] + 2*x[1] + 2)*(3*x[0]**2 + 6*x[0]*x[1] - 14*x[0] + 3*x[1]**2 - 14*x[1] + 19)) - ((x[0] + x[1] + 1)**2*(3*x[0]**2 + 6*x[0]*x[1] - 14*x[0] + 3*x[1]**2 - 14*x[1] + 19) + 1)*(576*x[1] - 384*x[0] - 432*x[0]*x[1] + 36*(2*x[0] - 3*x[1])**2 - (12*x[0] - 18*x[1])*(36*x[1] - 24*x[0] + 32) - (8*x[0] - 12*x[1])*(54*x[1] - 36*x[0] + 48) + 144*x[0]**2 + 324*x[1]**2 + 216) - ((12*x[0] - 18*x[1])*(12*x[0]**2 - 36*x[0]*x[1] - 32*x[0] + 27*x[1]**2 + 48*x[1] + 18) - (2*x[0] - 3*x[1])**2*(54*x[1] - 36*x[0] + 48))*((6*x[0] + 6*x[1] - 14)*(x[0] + x[1] + 1)**2 + (2*x[0] + 2*x[1] + 2)*(3*x[0]**2 + 6*x[0]*x[1] - 14*x[0] + 3*x[1]**2 - 14*x[1] + 19)) + ((2*x[0] - 3*x[1])**2*(12*x[0]**2 - 36*x[0]*x[1] - 32*x[0] + 27*x[1]**2 + 48*x[1] + 18) + 30)*(12*x[0]*x[1] - 28*x[1] - 28*x[0] + 2*(2*x[0] + 2*x[1] + 2)*(6*x[0] + 6*x[1] - 14) + 6*(x[0] + x[1] + 1)**2 + 6*x[0]**2 + 6*x[1]**2 + 38)],
                     [((8*x[0] - 12*x[1])*(12*x[0]**2 - 36*x[0]*x[1] - 32*x[0] + 27*x[1]**2 + 48*x[1] + 18) - (2*x[0] - 3*x[1])**2*(36*x[1] - 24*x[0] + 32))*((6*x[0] + 6*x[1] - 14)*(x[0] + x[1] + 1)**2 + (2*x[0] + 2*x[1] + 2)*(3*x[0]**2 + 6*x[0]*x[1] - 14*x[0] + 3*x[1]**2 - 14*x[1] + 19)) - ((x[0] + x[1] + 1)**2*(3*x[0]**2 + 6*x[0]*x[1] - 14*x[0] + 3*x[1]**2 - 14*x[1] + 19) + 1)*(576*x[1] - 384*x[0] - 432*x[0]*x[1] + 36*(2*x[0] - 3*x[1])**2 - (12*x[0] - 18*x[1])*(36*x[1] - 24*x[0] + 32) - (8*x[0] - 12*x[1])*(54*x[1] - 36*x[0] + 48) + 144*x[0]**2 + 324*x[1]**2 + 216) - ((12*x[0] - 18*x[1])*(12*x[0]**2 - 36*x[0]*x[1] - 32*x[0] + 27*x[1]**2 + 48*x[1] + 18) - (2*x[0] - 3*x[1])**2*(54*x[1] - 36*x[0] + 48))*((6*x[0] + 6*x[1] - 14)*(x[0] + x[1] + 1)**2 + (2*x[0] + 2*x[1] + 2)*(3*x[0]**2 + 6*x[0]*x[1] - 14*x[0] + 3*x[1]**2 - 14*x[1] + 19)) + ((2*x[0] - 3*x[1])**2*(12*x[0]**2 - 36*x[0]*x[1] - 32*x[0] + 27*x[1]**2 + 48*x[1] + 18) + 30)*(12*x[0]*x[1] - 28*x[1] - 28*x[0] + 2*(2*x[0] + 2*x[1] + 2)*(6*x[0] + 6*x[1] - 14) + 6*(x[0] + x[1] + 1)**2 + 6*x[0]**2 + 6*x[1]**2 + 38),  ((2*x[0] - 3*x[1])**2*(12*x[0]**2 - 36*x[0]*x[1] - 32*x[0] + 27*x[1]**2 + 48*x[1] + 18) + 30)*(12*x[0]*x[1] - 28*x[1] - 28*x[0] + 2*(2*x[0] + 2*x[1] + 2)*(6*x[0] + 6*x[1] - 14) + 6*(x[0] + x[1] + 1)**2 + 6*x[0]**2 + 6*x[1]**2 + 38) - 2*((12*x[0] - 18*x[1])*(12*x[0]**2 - 36*x[0]*x[1] - 32*x[0] + 27*x[1]**2 + 48*x[1] + 18) - (2*x[0] - 3*x[1])**2*(54*x[1] - 36*x[0] + 48))*((6*x[0] + 6*x[1] - 14)*(x[0] + x[1] + 1)**2 + (2*x[0] + 2*x[1] + 2)*(3*x[0]**2 + 6*x[0]*x[1] - 14*x[0] + 3*x[1]**2 - 14*x[1] + 19)) + ((x[0] + x[1] + 1)**2*(3*x[0]**2 + 6*x[0]*x[1] - 14*x[0] + 3*x[1]**2 - 14*x[1] + 19) + 1)*(864*x[1] - 576*x[0] - 648*x[0]*x[1] + 54*(2*x[0] - 3*x[1])**2 - 2*(12*x[0] - 18*x[1])*(54*x[1] - 36*x[0] + 48) + 216*x[0]**2 + 486*x[1]**2 + 324)])
                )


class BoothFunction(ObjectiveFunction):
	"""
	The Booth function is defined in Nocedal and Wright's
	textbook Numerical Optimization, Formula 2.23, on page 30.

	The formula has been rewritten slightly to account for the fact
	that the input array has two elements at indicies 0 and 1, whereas
	the formula defines two elements $x_1$ and $x_2$.  The reader should
	consider code $x_0$ = book $x_1$ and code $x_1$ = book $x_2$. 
	Therefore we write it as:

	$(2*x + y - 5)^2 + (x + 2*y - 7)^2$	
	"""

	def __init__(self):
		ObjectiveFunction.__init__(self)
		self.n = 2

	def descriptive_name(self):
		"""
		Return a descriptive name of the objective function, for
		use in the table headers.
		"""
		return 'the Booth function'

	def real_f(self, x):
		"""
		Returns the value of Booth function evaluated at $x$.
	
		Inputs:
			$x$ is a 2D Numeric array of floating point numbers which 
			represent the inputs to the Booth function.

		Outputs:
			Returns a floating point scalar with the value of the 
			Booth function evaluated at $x$.    

		Side Effects:
			None.
		"""
		return ((2*x[0] + x[1] - 5)**2 + (x[0] + 2*x[1] - 7)**2)

	def real_grad_f(self, x):
		"""
		Returns the value of the gradient of the Booth function 
		evaluated at $x$.  

		The gradient of the Booth function was calculated by hand
		to be:
		
		(10*x[0] + 8*x[1] - 34, 8*x[0] + 10*x[1] - 38)

		Inputs:
            	$x$ is a 2D Numeric array of floating point numbers 
				which represent the inputs to the Booth 
				gradient function
			
		Outputs:
			Returns a 2D Numeric array with the values of the 
			gradient of the Booth function evaluated at $x$.

		Side Effects:
			None.
	    """
		return array((10*x[0] + 8*x[1] - 34, 8*x[0] + 10*x[1] - 38))

	def real_hessian_f(self, x):
		"""
		Returns the value of the Hessian of the Booth function
		evaluated at $x$.

		The Hessian of the Booth function was calculated by hand
		to be:

		([10,  8],[8, 10])

		Inputs:
			$x$ is a 2D Numeric array of the floating point numbers
			which represent the inputs to the Booth Hessian
			function.

		Outputs:
			Returns a 2x2 Numeric matrix with the values of
			the Hessian of the Booth function evaluated at $x$.

		Side Effects:
			None.
		"""
		return array(([10,  8], [8, 10]))

class testObjFunc(ObjectiveFunction):
	"""
	A simple test Objective Function, with root
	at $x^\ast = (0,0)$, for testing the Line Search
	methods.
	
	$f(x) = x_0^2 + x_1^2$
	"""
	def descriptive_name(self):
		return 'test function'
	
	def real_f(self, x):
		return x[0]**2 + x[1]**2

	def real_grad_f(self, x):
		"""
		The gradient $\nabla f$ is calculated to be:
		
		$\nabla f = \left[ \begin{array}{c}
		   2x_0 \\ 2x_1 \end{array} \right]$
		"""
		return array((2.0*x[0], 2.0*x[1]))
	
	def real_hessian_f(self, x):
		"""
		The hessian $\nabla^2 f$ is calculated to be:

		$\nabla^2 f = \left[ \begin{array}{cc}
		  2 & 0 \\ 0 & 2 \end{array} \right]$
		"""
		return array(([2.0, 0.0], [0.0, 2.0]))

class testSecondFunc(ObjectiveFunction):
	"""
	A simple test objective function, with root at
	$x^\ast = (1,1)$ for testing the line search methods.
	This one is a bit more ill conditioned for a line search
	in that the variables do not contribute equally to 
	the solution of the problem.
	
	$f(x) = 250.0(x_0 - 1)^2 + 0.025*(x_1 - 1)^2$
	"""
	def descriptive_name(self):
		return 'a second test function'

	def real_f(self, x):
		"""
		Returns the function evaluated at $x$
		"""
		return 250.0*(x[0] - 1.0)**2 + 0.025*(x[1] - 1.0)**2
	
	def real_grad_f(self, x):
		"""
		The gradient $\nabla f$ is calculated to be:
		
		$\nabla f = \left[ \begin{array}{c}
		  500x_0 - 500 \\ 0.05x_1 - 0.05 \end{array} \right]$
		"""
		return array((500*x[0] - 500, 0.05*x[1] - 0.05))

	def real_hessian_f(self, x):
		"""
		The hessian $\nabla^2 f$ is calculated to be:

		$\nabla^2 f = \left[ \begin{array}{cc}
		  500 & 0 \\ 0 & 0.05 \end{array} \right]$
		"""
		return array(([500, 0.0], [0.0, 0.05]))


#####################
# unit testing code #
#####################
import unittest

class ObjectiveFunctionTestCase(unittest.TestCase):
	def testFunctionCaching(self):
		"""
		Check that the ObjectiveFunction class correctly
		counts the number of unique inputs to f
		"""
		o = ObjectiveFunction()
		
		assert len(o.f_cache) == 0, \
			'incorrect starting count for unique inputs'
		o.f(array((0.0, 0.0)))
		assert len(o.f_cache) == 1, \
			'incorrect count for one input'
		o.f(array((0.0, 0.0)))
		assert len(o.f_cache) == 1, \
			'incorrect count for one unique input'
		o.f(array((1.0, 0.0)))
		assert len(o.f_cache) == 2, \
			'incorrect count for two unique inputs'
		
		# check that the f cache didn't end up corrupting
		# the other caches
		assert len(o.f_grad_cache) == 0, \
			'f cache corrupted gradient cache'
		assert len(o.f_hessian_cache) == 0, \
			'f cache corrupted hessian cache'
		
		# check gradient cache
		o.grad_f(array((1.0, 0.0)))
		assert len(o.f_grad_cache) == 1, \
			'f grad cache not storing inputs'
		o.grad_f(array((1.0, 0.0)))
		assert len(o.f_grad_cache) == 1, \
			'f grad cache not storing unique inputs'
		
		# check Hessian cache
		o.hessian_f(array((0.0, 1.0)))
		assert len(o.f_hessian_cache) == 1, \
			'f hessian cache not storing inputs'
		o.hessian_f(array((0.0, 1.0)))
		assert len(o.f_hessian_cache) == 1, \
			'f hessian cache not storing unique inputs'

class RosenbrockTestCase(unittest.TestCase):
	def testRosenbrock(self):
		"""
		Check that the Rosenbrock function returns correct 
		output for certain inputs
		"""
		r = Rosenbrock()
		
		assert r.f(array((0.0, 0.0))) == 1, \
			'incorrect value returned for x=[0,0]'
		assert r.f(array((1.0, 0.0))) == 100, \
			'incorrect value returned for x=[1,0]'
		assert r.f(array((0.0, 1.0))) == 101, \
			'incorrect value returned for x=[0,1]'
		# check the case of the minimum of the Rosenbrock at
		# $x^\ast = (1,1)$
		assert r.f(array((1.0, 1.0))) == 0, \
			'incorrect value returned for x=[1,1]'


	def testRosenbrockGradient(self):
		"""
		Check that the Rosenbrock gradient function 
		returns correct output for certain inputs
		"""
		r = Rosenbrock()

		assert alltrue(r.grad_f(array((0.0, 0.0)))\
			       == array((-2.0, 0.0))), \
			'incorrect value returned for x=[0,0]'
		assert alltrue(r.grad_f(array((1.0, 0.0))) \
			       == array((400.0, -200.0))), \
			'incorrect value returned for x=[1,0]'
		assert alltrue(r.grad_f(array((0.0, 1.0))) \
			       == array((-2.0, 200.0))), \
			'incorrect value returned for x=[0,1]'
		# gradient should be zero at the minimum of the Rosenbrock at 
		# $x^\ast = (1,1)$
		assert alltrue(r.grad_f(array((1.0, 1.0))) == \
			       array((0.0, 0.0))), \
			'incorrect value returned for x=[1,1]'	

	def testRosenbrockHessian(self):
		"""
		Check that the Rosenbrock Hessian function
		returns correct output for certain inputs
		"""
		r = Rosenbrock()
		assert alltrue(r.hessian_f(array((0.0, 0.0))) \
			       == array(([2, 0],\
			[0, 200]))), 'incorrect value returned for x=[0,0]'
		assert alltrue(r.hessian_f(array((1.0, 0.0))) \
			       == array(([1202, -400],\
			[-400, 200]))), 'incorrect value returned for x=[1,0]'
		assert alltrue(r.hessian_f(array((0.0, 1.0))) \
			       == array(([-398, 0],\
			[0, 200]))), 'incorrect value returned for x=[0,1]'
		# Hessian should be pos def at the minimum of the Rosenbrock 
		# at $x^\ast= (1,1)$
		assert alltrue(r.hessian_f(array((1.0, 1.0))) \
			       == array(([802, -400],\
			[-400, 200]))), 'incorrect value returned for x=[1,1]'

class Problem4_1TestCase(unittest.TestCase):
	def testProblem4_1(self):
		"""
		Check that the Problem4_1 function returns correct 
		output for certain inputs
		"""
		p = Problem4_1()
		
		assert p.f(array((0.0, 0.0))) == 1, \
			'incorrect value returned for x=[0,0]'
		assert p.f(array((1.0, 0.0))) == 10, \
			'incorrect value returned for x=[1,0]'
		assert p.f(array((0.0, 1.0))) == 11, \
			'incorrect value returned for x=[0,1]'
		# check the case of the minimum of Problem4_1 at
		# $x^\ast = (1,1)$
		assert p.f(array((1.0, 1.0))) == 0, \
			'incorrect value returned for x=[1,1]'


	def testProblem4_1Gradient(self):
		"""
		Check that the Problem4_1 gradient function 
		returns correct output for certain inputs
		"""
		p = Problem4_1()

		assert alltrue(p.grad_f(array((0.0, 0.0))) == array((-2.0, 0.0))), \
			'incorrect value returned for x=[0,0]'
		assert alltrue(p.grad_f(array((1.0, 0.0))) == array((40.0, -20.0))), \
			'incorrect value returned for x=[1,0]'
		assert alltrue(p.grad_f(array((0.0, 1.0))) == array((-2.0, 20.0))), \
			'incorrect value returned for x=[0,1]'
		# gradient should be zero at the minimum of the Problem4_1 at 
		# $x^\ast = (1,1)$
		assert alltrue(p.grad_f(array((1.0, 1.0))) == array((0.0, 0.0))), \
			'incorrect value returned for x=[1,1]'	

	def testProblem4_1Hessian(self):
		"""
		Check that the Problem4_1 Hessian function
		returns correct output for certain inputs
		"""
		p = Problem4_1()

		assert alltrue(p.hessian_f(array((0.0, 0.0))) == array(([42, 0],\
			[0, 200]))), 'incorrect value returned for x=[0,0]'
		assert alltrue(p.hessian_f(array((1.0, 0.0))) == array(([42, -40],\
			[-40, 20]))), 'incorrect value returned for x=[1,0]'
		assert alltrue(p.hessian_f(array((0.0, 1.0))) == array(([2, 0],\
			[0, 20]))), 'incorrect value returned for x=[0,1]'
		# Hessian should be pos def at the minimum of the Problem4_1 
		# at $x^\ast= (1,1)$
		assert alltrue(p.hessian_f(array((1.0, 1.0))) == array(([2, -40],\
			[-40, 20]))), 'incorrect value returned for x=[1,1]'


class Problem4_3TestCase(unittest.TestCase):
	def testFunctionValues(self):
		pf = Problem4_3()
	
		assert pf.f(array((0.0, 0.0))) == 1, \
			'incorrect value returned for x=[0,0]'
		assert pf.f(array((1.0, 0.0))) == 10, \
			'incorrect value returned for x=[1,0]'
		assert pf.f(array((0.0, 1.0))) == 11, \
			'incorrect value returned for x=[0,1]'
		assert pf.f(array((1.0, 1.0))) == 0, \
			'incorrect value returned for x=[1,1]'

	def testFunctionGradientValues(self):
		pf = Problem4_3()
		
		assert alltrue(pf.grad_f(array((0.0, 0.0))) == array((-2.0, 0.0))), \
			'incorrect value returned for x=[0,0]'
		assert alltrue(pf.grad_f(array((1.0, 0.0))) == array((40.0, -20.0))), \
			'incorrect value returned for x=[1,0]'
		assert alltrue(pf.grad_f(array((0.0, 1.0))) == array((-2.0, 20.0))), \
			'incorrect value returned for x=[0,1]'
		assert alltrue(pf.grad_f(array((1.0, 1.0))) == array((0.0, 0.0))), \
			'incorrect value returned for x=[1,1]'

	def testFunctionHessianValues(self):
		pf = Problem4_3()
		
		assert alltrue(pf.hessian_f(array((0.0, 0.0))) == array(([2.0, 0.0],\
									 [0.0, 20.0]))), \
			'incorrect value returned for x=[0,0]'
		assert alltrue(pf.hessian_f(array((1.0, 0.0))) == array(([122.0, 0.0], \
									[0.0, 20.0]))), \
			 'incorrect value returned for x=[1,0]'
		assert alltrue(pf.hessian_f(array((0.0, 1.0))) == array(([-40.0, 0.0], \
									 [0.0, 20]))), \
			 'incorrect value returned for x=[0,1]'
		assert alltrue(pf.hessian_f(array((1.0, 1.0))) == array(([82.0, 0.0], \
									[0.0, 20.0]))), \
			 'incorrect value returned for x=[1,1]'

		pf4 = Problem4_3()
		assert alltrue(pf4.hessian_f(array((1.0, 1.0, 1.0, 1.0))) == \
			array(([82.0, 0.0,  0.0,  0.0],
				[0.0, 20.0,  0.0,  0.0],
				[0.0,  0.0, 82.0,  0.0],
				[0.0,  0.0,  0.0, 20.0]))), \
			 'incorrect value returned for x=[1,1,1,1]'
	
if __name__ == '__main__':
    unittest.main()
    
