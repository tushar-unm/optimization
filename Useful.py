from scipy import *
from scipy import linalg
#import numpy as np
import pandas as pd

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


def ConstructReportingDf(rep_dict, x_star, report_name=None, save=False):
	rep_df = None
	header = ['iteration', 'alpha_k', 'f_x_k', 'f_x_k_plus_1', 'norm_x_k_x_star', 'norm_x_k_plus_1_x_star', 'norm_grad_x_k', 'norm_grad_x_k_plus_1', 'conv_lin', 'conv_quad', 'conv_grad_lin', 'conv_grad_quad', 'storage', 'f_eval', 'grad_f_eval', 'hessian_f_eval']
	print(len(header))

	rep_ar = []
	for i in range(len(rep_dict)-1):
		temp = []
		temp.append(i)
		temp.append(rep_dict[i]['alpha_k'])
		temp.append(rep_dict[i]['f_x_k'])
		temp.append(rep_dict[i+1]['f_x_k'])
		norm_x_k_x_star = linalg.norm(rep_dict[i]['x_k'] - x_star)
		norm_x_k_plus_1_x_star = linalg.norm(rep_dict[i+1]['x_k'] - x_star)
		temp.append(norm_x_k_x_star)
		temp.append(norm_x_k_plus_1_x_star)
		temp.append(linalg.norm(rep_dict[i]['grad_f_x_k']))
		temp.append(linalg.norm(rep_dict[i+1]['grad_f_x_k']))
		temp.append(norm_x_k_plus_1_x_star/norm_x_k_x_star)
		temp.append(norm_x_k_plus_1_x_star/(norm_x_k_x_star*norm_x_k_x_star))
		temp.append(linalg.norm(rep_dict[i+1]['grad_f_x_k']) /
		            linalg.norm(rep_dict[i]['grad_f_x_k']))
		temp.append(linalg.norm(rep_dict[i+1]['grad_f_x_k']) /
		            (linalg.norm(rep_dict[i]['grad_f_x_k'])*linalg.norm(rep_dict[i]['grad_f_x_k'])))
		temp.append(rep_dict[i]['storage'])
		temp.append(rep_dict[i]['f_eval'])
		temp.append(rep_dict[i]['grad_f_eval'])
		temp.append(rep_dict[i]['hessian_f_eval'])
		rep_ar.append(temp)
	
	rep_df = pd.DataFrame(rep_ar, columns=header)
	if save:
		rep_df.to_csv('output/report_{}.csv'.format(report_name))
	return rep_df
