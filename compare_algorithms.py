#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 20:28:14 2017

@author: alain
"""

from time import time as time
from sklearn.metrics import precision_recall_fscore_support

import numpy as np
from screening_lasso import generate_random_gaussian
from noncvx_lasso import MMLasso, MMLasso_screening, \
    MMLasso_screening_genuine, BCD_noncvxlasso_lsp, check_opt_logsum, \
    approx_lsp
from noncvx_lasso import current_cost, reg_lsp, prox_lsp
from noncvx_lasso import GIST

import warnings
warnings.filterwarnings("ignore")

max_iter = 2000
maxiter_inner = 100
init_iter = 0



def run_BCD(X, y, lambd, p, tol=1e-5, w_init=[]):
    tic = time()
    w_bcd = BCD_noncvxlasso_lsp(X, y, lambd, p, tol=tol, max_iter=max_iter,
                                w_init=w_init)
    time_bcd = time() - tic
    return w_bcd, time_bcd


def run_GIST(X, y, lambd, p, tol=1e-5, w_init=[]):
    """
    generalized ISTA
    """
    tic = time()
    w_gist, _ = GIST(X, y, lambd, p, reg=reg_lsp, prox=prox_lsp, eta=1.5,
                     sigma=0.1, tol=tol, max_iter=max_iter, w_init=w_init)
    time_gist = time() - tic
    return w_gist, time_gist


def run_MMscreening(X, y, lambd, p, tol=1e-5, dual_gap_inner=1e-3,
                            w_init=[]):
    """
    majorization minimization only with inner screening
    """
    tic = time()
    w_mmlsp_screened_gen = MMLasso(X, y, lambd, p,
                                approx=approx_lsp,
                                maxiter=max_iter,
                                tol_first_order=tol,
                                dual_gap_inner=dual_gap_inner,
                                maxiter_inner=maxiter_inner,
                                w_init=w_init)
    time_mmlsp_screened_gen = time() - tic
    return w_mmlsp_screened_gen, time_mmlsp_screened_gen

def run_MMscreening_genuine(X, y, lambd, p, tol=1e-5, dual_gap_inner=1e-3,
                            w_init=[]):
    """
    majorization minimization only with inner screening
    """
    tic = time()
    w_mmlsp_screened_gen = MMLasso_screening_genuine(X, y, lambd, p,
                                                     approx=approx_lsp,
                                                     maxiter=max_iter,
                                                     tol_first_order=tol,
                                                     dual_gap_inner=dual_gap_inner,
                                                     maxiter_inner=maxiter_inner,
                                                     w_init=w_init)
    time_mmlsp_screened_gen = time() - tic
    return w_mmlsp_screened_gen, time_mmlsp_screened_gen


def run_MMscreening_2(X, y, lambd, p, tol=1e-5, dual_gap_inner=1e-3,
                      screen_frq=2, w_init=[]):
    """
    Majorization-minimization with inner screening and screening propagation
    """
    tic = time()
    w_mmlsp_screened = MMLasso_screening(X, y, lambd, p, approx=approx_lsp,
                                         maxiter=max_iter, initial_screen=True,
                                         method=2, screen_frq=screen_frq,
                                         tol_first_order=tol,
                                         dual_gap_inner=dual_gap_inner,
                                         maxiter_inner=maxiter_inner,
                                         w_init=w_init, algo_method='bcd',
                                         init_iter=init_iter)
    time_mmlsp_screened = time() - tic
    return w_mmlsp_screened, time_mmlsp_screened


def run_algo(X, y, lambd, p, algo, tol, dual_gap_inner=1e-3,
             screen_every=2, w_init=[]):

    if algo == 'bcd':
        w, run_time = run_BCD(X, y, lambd, p, tol=tol, w_init=w_init)
    elif algo == 'gist':
        w, run_time = run_GIST(X, y, lambd, p, tol=tol, w_init=w_init)
    elif algo == "no_screening":
        w, run_time = run_MMscreening(X, y, lambd, p,
                                              tol=tol,
                                              dual_gap_inner=dual_gap_inner,
                                              w_init=w_init)

    elif algo == 'mm_screening_genuine':
        w, run_time = run_MMscreening_genuine(X, y, lambd, p,
                                              tol=tol,
                                              dual_gap_inner=dual_gap_inner,
                                              w_init=w_init)

    elif algo == 'mm_screening_2':
        w, run_time = run_MMscreening_2(X, y, lambd, p,
                                        screen_frq=screen_frq, tol=tol,
                                        dual_gap_inner=dual_gap_inner,
                                        w_init=w_init)

    return w, run_time


def compute_performance(w, lambd, p, wopt, reg=reg_lsp, tol=1e-3):
    optimality = check_opt_logsum(X, y, w, lambd, p, tol=tol)
    maxi = np.max(abs(wopt - w))
    y_true = np.abs(wopt) > 0
    y_pred = np.abs(w) > 0
    f_meas_a = precision_recall_fscore_support(y_true, y_pred, pos_label=1,
                                               average='binary')[2]
    cost = current_cost(X, y, w, lambd, p, reg)
    return optimality, maxi, f_meas_a, cost

if __name__ == '__main__':
    
    n_iter = 1
    n_samplesvec =  [50]   # you can replace this value with 500 and
    n_featuresvec = [100]  # 5000 but i would take several days unless 
                           # you use one thread for one algo     
    n_informativevec = [5]

    sigma_bruit = 2        # settting 0.01, you can reproduce gain for bcd,
                           # MM_genuine and MM_screening 


    N = 100
    Tvec = np.power(10.0, -2 * np.linspace(1, N - 1, N) / (N - 1))
    pvec = [1]


    tolvec = [1e-3,1e-4,1e-5]
    dual_gap_inner = 1e-4
    screen_frq = 10
    path_results = './'
    algo_list = ['mm_screening_genuine', 'mm_screening_2']

    
    dual_gap_inner = 1e-4
    
    for tol in tolvec :
        for algo in algo_list:   
            print('running {}'.format(algo))
            for n_samples, n_features in zip(n_samplesvec, n_featuresvec):
                    for n_informative in n_informativevec:
                        filename = path_results  + '{}_n_samples{:d}_n_feat{:d}_n_inform{:d}_bruit{:2.2f}_N{:d}_tol{:2.5e}'.format(algo,n_samples,n_features,n_informative,sigma_bruit,N,tol)
                        opt_mm = 'gap_{:1.0e}_screen_{:d}'.format(dual_gap_inner,
                                                                  screen_frq)
                        filename = filename  + opt_mm +".txt"
                        filetxt = open(filename, "w")
                        filename = filename + opt_mm
                        print(filename)
                        timing = np.zeros([len(Tvec), len(pvec)])
                        optimality = np.zeros([len(Tvec), len(pvec)])
                        maxi = np.zeros([len(Tvec), len(pvec)])
                        f_meas = np.zeros([len(Tvec), len(pvec)])
                        cost = np.zeros([len(Tvec), len(pvec)])
        
                
                        X, y, wopt = generate_random_gaussian(n_samples,
                                                                n_features,
                                                                n_informative,
                                                                sigma_bruit)
                        lambdamax = np.max(np.abs(np.dot(X.transpose(), y)))
                        print("lamddaammax", lambdamax)
                        lambdavec = lambdamax * Tvec
                        # print(lambdavec)
                        for i_p, p in enumerate(pvec):
                            if i_p == 0:
                                w_init = []
                            else:
                                w_init = w_th.copy()
                            for i_lambd, lambd in enumerate(lambdavec):
                                print("lambda", lambd)
                                w, run_time = run_algo(X, y, lambd, p, algo,
                                                        tol, dual_gap_inner,
                                                        screen_frq, w_init)
                                w_init = w.copy()
                                if i_lambd == 0:
                                    w_th = w.copy()
                                timing[i_lambd, i_p] = run_time
                                tol_opt = max(tol, 1e-3)
                                optimality[i_lambd, i_p], maxi[i_lambd, i_p], f_meas[i_lambd, i_p], cost[i_lambd, i_p] = compute_performance(w,lambd, p,wopt,tol=tol_opt)
                                print(p, lambd, run_time,
                                        np.sum(timing[:, :]),
                                        f_meas[i_lambd, i_p],
                                        optimality[i_lambd, i_p],
                                        cost[i_lambd], i_p)
                                filetxt.write('{} {} {} {} {} {} {}\n'.format(
                                    p, 
                                    lambd, 
                                    run_time, 
                                    np.sum(timing[:, :]), 
                                    f_meas[i_lambd, i_p], 
                                    optimality[i_lambd, i_p],
                                    cost[i_lambd, i_p],
                                ))
                        #np.savez(filename, timing=timing, optimality=optimality,
                        #         maxi=maxi, f_meas=f_meas, cost=cost)
