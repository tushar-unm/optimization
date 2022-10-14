from LineSearch import *
#from TrustRegion import *
from ObjectiveFunctions import *
#from ConjugateGradient import *
from scipy import *
from Useful import *
import code as c

def prob_5_a():
    # x_0 = array((5.8, 1.3))
    # alpha_0 = 0.2
    # x_star = array((1.0,3.0))
    
    # rep_dict = NewtonsMethodLineSearch(x_0, alpha_0,\
	# 		   BacktrackingAlpha, \
	# 		   BoothFunction(), \
	# 		   9.9e-10)

    #x_0 = array((1, 1))
    # x_0 = array((1.8, 0.1))
    # alpha_0 = 0.1
    # x_star = array((0.0, -1.0))

    # rep_dict = NewtonsMethodLineSearch(x_0, alpha_0,\
	# 		   BacktrackingAlpha, \
	# 		   GoldsteinPriceFunction(), \
	# 		   9.9e-10)

    #x_0 = array((1, 2))
    # x_0 = array((0.0, 0.0))
    # x_0 = array((-1.0, 1.0))
    # alpha_0 = 0.1
    # x_star = array((3.0, 0.5))

    # rep_dict = NewtonsMethodLineSearch(x_0, alpha_0,\
	# 		   BacktrackingAlpha, \
	# 		   BealesFunction(), \
	# 		   9.9e-10)

    # x_0 = array((5.0, 3.0))
    # alpha_0 = 0.1
    # x_star = array((0.0, 0.0))

    # rep_dict = NewtonsMethodLineSearch(x_0, alpha_0,\
	# 		   BacktrackingAlpha, \
	# 		   SphereFunction(), \
	# 		   9.9e-10)

    
    
    rep_df = ConstructReportingDf(rep_dict, x_star, report_name='first_test')

    print(rep_df)

    return rep_df


def prob_5_b():
    # x_0 = array((6.2, 8.3))
    # x_0 = array((5.8, 1.3))
    # x_0 = array((3.5, 0.4))
    # alpha_0 = 0.1
    # x_star = array((1.0,3.0))

    # rep_dict = SteepestDescentLineSearch(x_0, alpha_0,\
	# 		   BacktrackingAlpha, \
	# 		   BoothFunction(), \
	# 		   9.9e-20)
    # rep_df = ConstructReportingDf(
    #     rep_dict, x_star, report_name='steepest_descent_booth_pt_3', save=True)

    #x_0 = array((1, 1))
    #x_0 = array((-1.2, 0.7))
    # x_0 = array((1.3, -0.4))
    # alpha_0 = 0.01
    # x_star = array((0.0, -1.0))

    # rep_dict = SteepestDescentLineSearch(x_0, alpha_0,\
	# 		   BacktrackingAlpha, \
	# 		   GoldsteinPriceFunction(), \
	# 		   9.9e-10)
    # rep_df = ConstructReportingDf(
    #     rep_dict, x_star, report_name='steepest_descent_goldstein_price_pt_3', save=True)

    #x_0 = array((1, 2))
    #x_0 = array((0.0, 0.0))
    #x_0 = array((-1.0, -1.0))
    # alpha_0 = 0.1
    # x_star = array((3.0, 0.5))

    # rep_dict = SteepestDescentLineSearch(x_0, alpha_0,\
	# 		   BacktrackingAlpha, \
	# 		   BealesFunction(), \
	# 		   9.9e-10)
    # rep_df = ConstructReportingDf(
    #     rep_dict, x_star, report_name='steepest_descent_beales_pt_3', save=True)

    # x_0 = array((5.0, 3.0))
    #x_0 = array((-4.5, -7.2))
    x_0 = array((15.1, -23.7))
    alpha_0 = 0.1
    x_star = array((0.0, 0.0))

    rep_dict = SteepestDescentLineSearch(x_0, alpha_0,\
			   BacktrackingAlpha, \
			   SphereFunction(), \
			   9.9e-20)

    rep_df = ConstructReportingDf(rep_dict, x_star, report_name='steepest_descent_sphere_pt_3', save=True)

    print(rep_df)

    return rep_df


def prob_5_c():
    #x_0 = array((6.2, 8.3))
    x_0 = array((5.8, 1.3))
    #x_0 = array((3.5, 0.4))
    alpha_0 = 0.1
    x_star = array((1.0,3.0))
    b = BoothFunction()

    rep_dict = BFGSLineSearch(x_0, alpha_0,\
			   BacktrackingAlpha, \
			   b, \
			   9.9e-10, \
               array(([1,0],[0,1])))#linalg.inv(b.hessian_f(x_0)))
    rep_df = ConstructReportingDf(
        rep_dict, x_star, report_name='bfgs_booth_pt_1', save=True)

    #x_0 = array((1, 1))
    #x_0 = array((-1.2, 0.7))
    # x_0 = array((1.3, -0.4))
    # alpha_0 = 0.01
    # x_star = array((0.0, -1.0))

    # rep_dict = BFGSLineSearch(x_0, alpha_0,\
	# 		   BacktrackingAlpha, \
	# 		   GoldsteinPriceFunction(), \
	# 		   9.9e-10)
    # rep_df = ConstructReportingDf(
    #     rep_dict, x_star, report_name='bfgs_goldstein_price_pt_3', save=True)

    #x_0 = array((1, 2))
    #x_0 = array((0.0, 0.0))
    #x_0 = array((-1.0, -1.0))
    # alpha_0 = 0.1
    # x_star = array((3.0, 0.5))

    # rep_dict = BFGSLineSearch(x_0, alpha_0,\
	# 		   BacktrackingAlpha, \
	# 		   BealesFunction(), \
	# 		   9.9e-10)
    # rep_df = ConstructReportingDf(
    #     rep_dict, x_star, report_name='bfgs_beales_pt_3', save=True)

    # # x_0 = array((5.0, 3.0))
    # # x_0 = array((-4.5, -7.2))
    # x_0 = array((15.1, -23.7))
    # alpha_0 = 0.1
    # x_star = array((0.0, 0.0))

    # rep_dict = BFGSLineSearch(x_0, alpha_0,
    #                           BacktrackingAlpha,
    #                           SphereFunction(),
    #                           9.9e-20)

    # rep_df = ConstructReportingDf(
    #     rep_dict, x_star, report_name='bfgs_sphere_pt_3', save=True)

    print(rep_df)

    return rep_df

if __name__ == '__main__':
    #res = prob_5_a()
    # res = prob_5_b()
    res = prob_5_c()
