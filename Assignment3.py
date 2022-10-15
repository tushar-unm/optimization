from LineSearch import *
from TrustRegion import *
from ObjectiveFunctions import *
#from ConjugateGradient import *
from scipy import *
from Useful import *
import code as c

# Newtons method
def prob_5_a():
    #x_0 = array((6.2, 8.3))
    #x_0 = array((5.8, 1.3))
    #x_0 = array((3.5, 0.4))
    # alpha_0 = 0.5
    # x_star = array((1.0,3.0))

    # rep_dict = NewtonsMethodLineSearch(x_0, alpha_0,\
	# 		   BacktrackingAlpha, \
	# 		   BoothFunction(), \
	# 		   9.9e-20)
    # rep_df = ConstructReportingDf(
    #     rep_dict, x_star, report_name='newtons_method_booth_pt_3', save=True)
    # print(rep_df)

    #x_0 = array((1, 1))
    #x_0 = array((-1.2, 0.7)) # converge for aplha=0.1 & 0.1
    #x_0 = array((1.3, -0.4)) #did not converge
    #x_0 = array((-1.3, -0.4)) # did not converge for difff alpha initially, converge for alpha=1
    # x_0 = array((0.5, 0.9)) #did not converge
    #x_0 = array((1.5, 1.9)) # worked for alpha=0.1, 1
    # alpha_0 = 1
    # x_star = array((0.0, -1.0))

    # rep_dict = NewtonsMethodLineSearch(x_0, alpha_0,\
	# 		   BacktrackingAlpha, \
	# 		   GoldsteinPriceFunction(), \
	# 		   9.9e-15)
    # rep_df = ConstructReportingDf(
    #     rep_dict, x_star, report_name='newtons_method_goldstein_price_pt_1', save=True)
    # print(rep_df)

    # x_0 = array((1, 2))
    #x_0 = array((0.0, 0.0))
    #x_0 = array((0.2, 4.9)) # original pt 3 x_0 = array((-1, -1)) which did not converge
    # alpha_0 = 0.1
    # x_star = array((3.0, 0.5))

    # rep_dict = NewtonsMethodLineSearch(x_0, alpha_0,\
	# 		   BacktrackingAlpha, \
	# 		   BealesFunction(), \
	# 		   9.9e-20)
    # rep_df = ConstructReportingDf(
    #     rep_dict, x_star, report_name='newtons_method_beales_pt_3', save=True)
    # print(rep_df)

    #x_0 = array((5.0, 3.0))
    #x_0 = array((-4.5, -7.2))
    # x_0 = array((15.1, -23.7))
    # alpha_0 = 0.75
    # x_star = array((0.0, 0.0))

    # rep_dict = NewtonsMethodLineSearch(x_0, alpha_0,\
	# 		   BacktrackingAlpha, \
	# 		   SphereFunction(), \
	# 		   9.9e-20)

    # rep_df = ConstructReportingDf(rep_dict, x_star, report_name='newtons_method_sphere_pt_3', save=True)

    # print(rep_df)

    return rep_df

#Steepest Descent
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
    # x_0 = array((15.1, -23.7))
    # alpha_0 = 0.1
    # x_star = array((0.0, 0.0))

    # rep_dict = SteepestDescentLineSearch(x_0, alpha_0,
    #                                      BacktrackingAlpha,
    #                                      SphereFunction(),
    #                                      9.9e-20)

    # rep_df = ConstructReportingDf(
    #     rep_dict, x_star, report_name='steepest_descent_sphere_pt_3', save=True)

    print(rep_df)

    return rep_df

#BFGS
def prob_5_c():
    # x_0 = array((6.2, 8.3))
    # #x_0 = array((5.8, 1.3))
    # #x_0 = array((3.5, 0.4))
    # alpha_0 = 0.1
    # x_star = array((1.0,3.0))
    # b = BoothFunction()

    # rep_dict = BFGSLineSearch(x_0, alpha_0,\
	# 		   BacktrackingAlpha, \
	# 		   b, \
	# 		   9.9e-7, \
    #            array(([1,0],[0,1])))#linalg.inv(b.hessian_f(x_0)))
    # rep_df = ConstructReportingDf(
    #     rep_dict, x_star, report_name='bfgs_booth_pt_1', save=True)
    # print(rep_df)

    #x_0 = array((1, 1))
    #x_0 = array((-1.2, 0.7))
    #x_0 = array((1.3, -0.4))
    # alpha_0 = 0.01
    # x_star = array((0.0, -1.0))

    # rep_dict = BFGSLineSearch(x_0, alpha_0,\
	# 		   BacktrackingAlpha, \
	# 		   GoldsteinPriceFunction(), \
	# 		   9.9e-8, \
    #             array(([1, 0], [0, 1])))
    # rep_df = ConstructReportingDf(
    #     rep_dict, x_star, report_name='bfgs_goldstein_price_pt_1', save=True)
    # print(rep_df)

    #x_0 = array((1, 2))
    #x_0 = array((0.0, 0.0))
    # x_0 = array((-1.0, -1.0))
    # alpha_0 = 0.1
    # x_star = array((3.0, 0.5))

    # rep_dict = BFGSLineSearch(x_0, alpha_0,\
	# 		   BacktrackingAlpha, \
	# 		   BealesFunction(), \
	# 		   9.9e-10)
    # rep_df = ConstructReportingDf(
    #     rep_dict, x_star, report_name='bfgs_beales_pt_3', save=True)

    # x_0 = array((5.0, 3.0))
    # #x_0 = array((-4.5, -7.2))
    # #x_0 = array((15.1, -23.7))
    # alpha_0 = 0.1
    # x_star = array((0.0, 0.0))

    # rep_dict = BFGSLineSearch(x_0, alpha_0,
    #                           BacktrackingAlpha,
    #                           SphereFunction(),
    #                           9.9e-20,
    #                           array(([1,0],[0,1])))
    # rep_df = ConstructReportingDf(
    #     rep_dict, x_star, report_name='bfgs_sphere_pt_1', save=True)

    #print(rep_df)

    return rep_df

#CG Steihaug
def prob_6():
    # x_0 = array((0.0, 0.0))
    # #x_0 = array((5.8, 1.3))
    # #x_0 = array((3.5, 0.4))
    # p = BoothFunction()
    # B_k = p.hessian_f(x_0)
    # epsilon = 9.9e-10
    # g = p.grad_f(x_0)
    # x_star = array((1.0,3.0))
    # delta = 0.1

    # rep_dict = CG_Steihaug(p, epsilon, delta, B_k, g)
    # rep_df = ConstructReportingDfCGSteihaug(
    #     rep_dict, x_star, x_0, report_name='cgsteihaug_booth_pt_1', save=True)
    # print(rep_df)

    # x_0 = array((1, 1))
    # x_0 = array((-1.2, 0.7))
    # x_0 = array((1.3, -0.4))
    # p = GoldsteinPriceFunction(2)
    # B_k = p.hessian_f(x_0)
    # epsilon = 9.9e-13
    # g = p.grad_f(x_0)
    # x_star = array((0.0, -1.0))
    # delta = 0.1

    # rep_dict = CG_Steihaug(p, epsilon, delta, B_k, g)
    # rep_df = ConstructReportingDfCGSteihaug(
    #     rep_dict, x_star, x_0, report_name='cgsteihaug_goldstein_price_pt_1', save=True)
    # print(rep_df)

    #x_0 = array((1, 2))
    x_0 = array((4.0, 3.0))
    # x_0 = array((-1.0, -1.0))
    p = BealesFunction()
    B_k = p.hessian_f(x_0)
    epsilon = 9.9e-5
    g = p.grad_f(x_0)
    x_star = array((3.0, 0.5))
    delta = 100

    rep_dict = CG_Steihaug(p, epsilon, delta, B_k, g)
    rep_df = ConstructReportingDfCGSteihaug(
        rep_dict, x_star, x_0, report_name='cgsteihaug_beales_pt_3', save=True)

    # x_0 = array((2.0, 3.0))
    # #x_0 = array((-4.5, -7.2))
    # #x_0 = array((15.1, -23.7))
    # p = SphereFunction()
    # B_k = p.hessian_f(x_0)
    # epsilon = 9.9e-10
    # g = p.grad_f(x_0)
    # x_star = array((0.0, 0.0))
    # delta =  10

    # rep_dict = CG_Steihaug(p, epsilon, delta, B_k, g)
    # rep_df = ConstructReportingDfCGSteihaug(
    #     rep_dict, x_star, x_0, report_name='cgsteihaug_sphere_pt_1', save=True)

    # # print(rep_df)

    # return rep_df


if __name__ == '__main__':
    #res = prob_5_a()
    # res = prob_5_b()
    #res = prob_5_c()
    res = prob_6()
