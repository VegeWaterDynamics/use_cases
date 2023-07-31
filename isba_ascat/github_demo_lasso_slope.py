#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 15:32:18 2023

@author: xushan
"""

import pickle
import pandas as pd
import logging
import sys
import numpy as np
# from ml_lsmodel_ascat.dnn import NNTrain
# from ml_lsmodel_ascat.jackknife import JackknifeGPI
import shap
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from sklearn.linear_model import LassoCV, RidgeCV, MultiTaskLassoCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import scipy
# from tensorflow import keras
# from tensorflow.keras import layers
# import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_probability as tfp
import seaborn as sns
from matplotlib import cm
import scipy.stats as st
import datetime
from ml_lsmodel_ascat.plot import plot_gsdata, plot_tsdata
from ml_lsmodel_ascat.util import performance, normalize
#%%
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    stream=sys.stdout)
def jacobian(temp_data, model, input_var_list_prac, output_var_list,
             output_var_list_pre, obs_var_list
             ):
    exp_name = '/before_stand_epsilon_001/'
    starttime = datetime.datetime.now()
    
    intep_method = "sc"
    input_var_list = [
        'WG1','WG2', 'WG3','WG4','WG5', 'WG6','WG7','WG8', 'WG9','WG10',
        'RN_ISBA', 'GPP_ISBA', 'LETR_ISBA', 'WR_ISBA',
        'LAI_ISBA', 'XRS_ISBA']
    GPI = ['Broadleaf', 'Agriculture','Grassland', 'Needleleaf']
#    year = 2019
    year = "climatology"
    
    ls_plot_shap, title_list = [],[]
    ls_plot_shap_med = []
    season_list = ["all"]
    possible_input_size = 20
    slope_season_list = []
    
    season = "all"
    jacob_gpi_list, nsc_gpi_list = [],[]
    
    if 1:
        i=0
        # temp_data = max_frac_df['data'].iloc[i].copy()
        one_gpi = temp_data
        
        input_var_list_prac_temp = input_var_list_prac#[0:10]
        # output_var_list = ['observed_sig', 'observed_slope']
        # output_var_list_pre = ['predicted_sig','predicted_slope']
        # obs_var_list = ['sig', 'slop']
        ###########
        # calculate the estimated sig & slope
        one_gpi[output_var_list] = one_gpi[obs_var_list]
        
        gpi_input, scaler_input = normalize(
                one_gpi[input_var_list_prac], 'standard')
        gpi_output, scaler_output = normalize(
                one_gpi[output_var_list], 'standard')
        
        predicted = model.predict(gpi_input)
        re_predicted = scaler_output.inverse_transform(predicted.reshape(-1,1))
        
        for i_obs, obs in enumerate(output_var_list_pre):
            one_gpi[obs] = re_predicted[:, i_obs]
        # one_gpi['predicted_sig'] = re_predicted[:, 0]
        # one_gpi['predicted_slope'] = re_predicted[:, 1]
        ###########
        
        if season == "all":
            temp_input = temp_data
            temp_input_all = temp_input.copy()
        
#        df_long.iloc[0].data[output_var_list] = df_long.iloc[0].data[obs_var_list]
#        df_long.iloc[0].data[output_var_list_pre] = one_gpi[output_var_list_pre]
        # max_frac_df.iloc[0].data[output_var_list] = max_frac_df.iloc[0].data[obs_var_list]
        # max_frac_df.iloc[0].data[output_var_list_pre] = one_gpi[output_var_list_pre]
        temp_data[output_var_list] = temp_data[obs_var_list]
        temp_data[output_var_list_pre] = one_gpi[output_var_list_pre]
        
#        print(df_long.iloc[0].data[output_var_list_pre])
        #%
        gpi_input, scaler_input = normalize(
                one_gpi[input_var_list_prac], 'standard')
        gpi_output, scaler_output = normalize(
                one_gpi[output_var_list], 'standard')
        
        bench_mark = (temp_input[input_var_list_prac] - scaler_input.mean_)/scaler_input.scale_
        predicted_bm = model.predict(bench_mark)
        re_predicted_bm = scaler_output.inverse_transform(predicted_bm.reshape(-1,1))
        
        df_jacob = pd.DataFrame()
        df_nsc = pd.DataFrame()
#        output_var = ['sig', 'slope', 'curv']
        # output_var = ['sig', 'slope']
        output_var = obs_var_list
        perturb_percent = 0.05
        slope_list = []
        
        list_perturb_stand = []
        list_perturb = []
        matrix_perturb_stand = np.zeros([len(input_var_list_prac_temp),
                                         possible_input_size,
                                         len(temp_input_all),
                                         len(input_var_list_prac)]
                                         )
        matrix_perturb = np.zeros([len(input_var_list_prac_temp),
                                   possible_input_size,
                                   len(temp_input_all),
                                   len(input_var_list_prac)]
                                         )
        
#        print(temp_input_all.columns)
#        print(temp_input_all[output_var_list_pre])
#        print(max_frac_df.iloc[0].data[output_var_list_pre])
        
        for j, x in enumerate(input_var_list_prac_temp):
#        if 1:
#            j=9
            if season == "all":
                #%
                # loop over the possible input size
                range_x = temp_data[x].max() - temp_data[x].min()
                perturb_amount = perturb_percent * range_x
                perturb_amount_arr = np.arange(-perturb_amount,
                                              perturb_amount,
                                              perturb_amount*2/possible_input_size)
                for n in range(possible_input_size):
                    matrix_perturb[j,n,:,:] = temp_input_all[input_var_list_prac].values
                    matrix_perturb[j,n,:,j] = temp_input_all[x] + perturb_amount_arr[n]
                    
                    matrix_perturb_stand[j,n,:,:] = gpi_input
                    x_pert_stand = (temp_input_all[x]+perturb_amount_arr[n]- scaler_input.mean_[j])/scaler_input.scale_[j]
                    matrix_perturb_stand[j,n,:,j] = x_pert_stand
        model_input = matrix_perturb_stand.reshape((-1,len(input_var_list_prac)))
        model_output = model.predict(model_input)
        re_model_output = scaler_output.inverse_transform(model_output.reshape(-1,1))
        model_output_1 = re_model_output.reshape((len(input_var_list_prac_temp),
                                                possible_input_size,
                                                len(temp_input_all),
#                                                3))
                                                len(output_var)))
#        model_output_1 = re_model_output.reshape((len(list_perturb_stand),
#                                                possible_input_size,
#                                                3))
        for j, x in enumerate(input_var_list_prac_temp):
            if season == "all":
                #%
                for m in range(len(output_var)):
                    jacob_list = []
                    nsc_list = []
                    for k in range(len(temp_input_all)):
#                        m=0
#                        perturb_arr_x = list_perturb[j*len(temp_input_all)+k][:,j]
#                        re_predicted_pert = model_output_1[j*len(temp_input_all)+k,
#                                                           :,:]
                        perturb_arr_x = matrix_perturb[j,:,k,j]
                        re_predicted_pert = model_output_1[j,:,k,:]
                        slope, intercept, r_value, p_value, std_err = st.linregress(perturb_arr_x,
                                                                                    re_predicted_pert[:,m])
                        jacob_list.append(slope)
                        # nsc = slope * temp_input_all[x].iloc[k]/np.abs(temp_input_all.iloc[k][output_var_list_pre[m]])
                        nsc = slope * temp_input_all[x].iloc[k]/np.abs(temp_input_all.iloc[k][output_var_list[m]])
                        nsc_list.append(nsc)
#                        print(k)
                    df_jacob[x+'_'+output_var[m]] = jacob_list
                    df_nsc[x+'_'+output_var[m]] = nsc_list
                df_jacob.index = temp_data.index
                df_jacob_median_doy = df_jacob.groupby(df_jacob.index.dayofyear).median()

                df_nsc.index = temp_data.index
                df_nsc_median_doy = df_nsc.groupby(df_nsc.index.dayofyear).median()
        endtime = datetime.datetime.now()
        print((endtime - starttime).seconds)
        print(i)
        jacob_gpi_list.append(df_jacob)
        nsc_gpi_list.append(df_nsc)
    # max_frac_df['jacob'] = jacob_gpi_list
    # max_frac_df['nsc'] = nsc_gpi_list
    return df_jacob, df_nsc
def shap_values(model, input_whole):
    background = input_whole[np.random.choice(input_whole.shape[0],
                                              1000,
                                              replace=False)]
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(input_whole)

    return shap_values
def lasso_cv_train(gpi_data, input_list, output_list, val_split_year):
    if len(gpi_data) > 0:
        alphas = np.logspace(-4, -0.5, 30)
        df_input_output = gpi_data[input_list+output_list].copy()
        X = df_input_output[input_list][df_input_output.index.year < val_split_year]
        X_norm = (X - X.mean())/X.std()
        y = df_input_output[output_list][df_input_output.index.year < val_split_year]
        y_norm = (y - y.mean())/y.std()
        
        year_list = df_input_output[df_input_output.index.year < val_split_year].index.year.unique()
        k_fold = KFold(len(year_list), shuffle=True, 
                   random_state=0)
        alpha_opt = 0
        lasso_opt = 0
        k_cv = 0
        score_opt = 10
        lasso_cv_list = []
        alpha_list = []
        
        for k, (train, test) in enumerate(k_fold.split(X_norm, y_norm)):
            # lasso_cv = LassoCV(alphas=alphas, random_state=0, max_iter=1000000)
            lasso_cv = MultiTaskLassoCV(alphas=alphas, random_state=0, max_iter=1000000)
            lasso_cv.fit(X_norm.values[train], y_norm.values[train])
            score_k = mean_squared_error(
                            lasso_cv.predict(X_norm.values[test]) * y.std().values+y.mean().values, 
                            y_norm.values[test] * y.std().values+y.mean().values,
                            squared=False)
            print(
                "[fold {0}] alpha: {1:.5f}, score: {2:.5f}".format(
                    k, lasso_cv.alpha_, 
                    score_k
                )
            )
            alpha_list.append(lasso_cv.alpha_)
            print(test)
            print(lasso_cv.coef_)
            lasso_cv_list.append(lasso_cv)
            if score_k <= score_opt:
                alpha_opt = lasso_cv.alpha_
                lasso_opt = lasso_cv
                score_opt = score_k
                k_cv = k
                
        print("Answer: Not very much since we obtained different alphas for different")
        print("subsets of the data and moreover, the scores for these alphas differ")
        print("quite substantially.")
        
        df = pd.DataFrame()
        df_perf = pd.DataFrame()
        X_all = df_input_output[input_list]
        X_all_norm = (X_all - X.mean())/X.std()
        lasso_coef_all = {}
        # df['predicted_slope (Lasso Regression opt)'] = lasso_opt.predict(X_all_norm) * y.std()+y.mean()
        for i_obs, obs_var in enumerate(output_list):
            tmp = lasso_opt.predict(X_all_norm) * y.std().values+y.mean().values
            df['predicted_'+obs_var+' (Lasso Regression opt)'] = tmp[:, i_obs]
            df['observed_'+obs_var] = df_input_output[obs_var].values
            df.index = df_input_output.index
            i_model = 0
            bias_list, rho_list, ubrmse_list = [],[],[]
            lasso_cv_coef_list = []
            pd_lasso_coef = pd.DataFrame()
            for lasso_cv in lasso_cv_list:
                tmp_cv = lasso_cv.predict(X_all_norm) * y.std().values+y.mean().values
                df['predicted_'+obs_var+' (Lasso Regression '+str(i_model)+')'] = tmp_cv[:, i_obs]
                ubrmse = mean_squared_error(df[df.index.year>=val_split_year+1]['predicted_'+obs_var+' (Lasso Regression '+str(i_model)+')'].dropna() -\
                                    df[df.index.year>=val_split_year+1]['predicted_'+obs_var+' (Lasso Regression '+str(i_model)+')'].dropna().mean(), 
                                 df[df.index.year>=val_split_year+1]['observed_'+obs_var].dropna() -\
                                    df[df.index.year>=val_split_year+1]['observed_'+obs_var].dropna().mean(), squared=False)
                rho = scipy.stats.pearsonr(df[df.index.year>=val_split_year+1]['predicted_'+obs_var+' (Lasso Regression '+str(i_model)+')'].dropna(),
                                           df[df.index.year>=val_split_year+1]['observed_'+obs_var].dropna())
                bias = np.mean(df[df.index.year>=val_split_year+1]['predicted_'+obs_var+' (Lasso Regression '+str(i_model)+')'].dropna() - \
                              df[df.index.year>=val_split_year+1]['observed_'+obs_var].dropna())
                pd_lasso_coef[str(i_model)] = lasso_cv.coef_[i_obs,:]
                ubrmse_list.append(ubrmse)
                rho_list.append(rho[0])
                bias_list.append(bias)
                
                i_model += 1
            pd_lasso_coef['mean'] = pd_lasso_coef.mean(axis = 1)
            pd_lasso_coef['std'] = pd_lasso_coef.std(axis = 1)
            lasso_coef_all[obs_var] = pd_lasso_coef
            df_perf['ubrmse_'+obs_var] = ubrmse_list
            df_perf['bias_'+obs_var] = bias_list
            df_perf['rho_'+obs_var] = rho_list
    return lasso_cv_list, df_perf, lasso_coef_all
def RandomForestRegressor_cv_train(gpi_data, input_list, 
                                   output_list, val_split_year,
                                   jacobian_or_nsc):
    if len(gpi_data) > 0:
        alphas = np.logspace(-4, -0.5, 30)
        df_input_output = gpi_data[input_list+output_list].copy()
        X = df_input_output[input_list][df_input_output.index.year < val_split_year]
        X_norm = (X - X.mean())/X.std()
        y = df_input_output[output_list][df_input_output.index.year < val_split_year]
        y_norm = (y - y.mean())/y.std()
        
        year_list = df_input_output[df_input_output.index.year < val_split_year].index.year.unique()
        k_fold = KFold(len(year_list), shuffle=True, 
                   random_state=0)
        alpha_opt = 0
        lasso_opt = 0
        k_cv = 0
        score_opt = 10
        regr_cv_list = []
        alpha_list = []
        
        for k, (train, test) in enumerate(k_fold.split(X_norm, y_norm)):
            # lasso_cv = LassoCV(alphas=alphas, random_state=0, max_iter=1000000)
            # lasso_cv = MultiTaskLassoCV(alphas=alphas, random_state=0, max_iter=1000000)
            regr_cv = RandomForestRegressor(max_depth=10, random_state=0)
            regr_cv.fit(X_norm.values[train], y_norm.values[train])
            score_k = mean_squared_error(
                            regr_cv.predict(X_norm.values[test]) * y.std().values+y.mean().values, 
                            y_norm.values[test] * y.std().values+y.mean().values,
                            squared=False)
            print(
                "[fold {0}] score: {1:.5f}".format(
                    k, 
                    score_k
                )
            )
            print(test)
            regr_cv_list.append(regr_cv)
            if score_k <= score_opt:
                regr_opt = regr_cv
                score_opt = score_k
                k_cv = k
                
        print("Answer: Not very much since we obtained different alphas for different")
        print("subsets of the data and moreover, the scores for these alphas differ")
        print("quite substantially.")
        
        X_all = df_input_output[input_list]
        X_all_norm = (X_all - X.mean())/X.std()

        fig, ax = plt.subplots(figsize=(20,4))
        i_model = 0
        df = pd.DataFrame()
        df['observed_'+output_list[0]] = df_input_output[output_list]
        
        for regr_cv in regr_cv_list:
            df['predicted_slope (RF Regression '+str(i_model)+')'] = regr_cv.predict(X_all_norm) * y.std().values+y.mean().values
            df_plot = df#pd.concat([df, df_time], axis=1)
            i_model += 1
        
        
        
        i=0
        colors = ['r']+list(cm.Greens(np.linspace(0, 1, len(regr_cv_list)+3))[::-1])
        style_list = ['-']*len(regr_cv_list)
        linewidths = [1]*len(regr_cv_list)
        label_list = df_plot.columns#[0:-1]
    
        for col, style,color, lw,label in zip(df_plot.columns, style_list,
                                        colors, linewidths,label_list):
            df_plot[df_plot.index.year>=val_split_year][col].plot(color=color, style=style, markersize=2,lw=lw, ax=ax,label=label).legend(loc='upper right')
            i+=1
        ax.set_ylabel(output_list[0], fontsize=13)
        plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)
        
        df = pd.DataFrame()
        df_perf = pd.DataFrame()
        X_all = df_input_output[input_list]
        X_all_norm = (X_all - X.mean())/X.std()
        regr_coef_all = {}
        # df['predicted_slope (Lasso Regression opt)'] = lasso_opt.predict(X_all_norm) * y.std()+y.mean()
        for i_obs, obs_var in enumerate(output_list):
            tmp = regr_opt.predict(X_all_norm) * y.std().values+y.mean().values
            df['predicted_'+obs_var+' (RF Regression opt)'] = tmp#[:, i_obs]
            df['observed_'+obs_var] = df_input_output[obs_var].values
            df.index = df_input_output.index
            i_model = 0
            bias_list, rho_list, ubrmse_list = [],[],[]
            regr_cv_coef_list = []
            pd_regr_coef = pd.DataFrame()
            for regr_cv in regr_cv_list:
                tmp_cv = regr_cv.predict(X_all_norm) * y.std().values+y.mean().values
                df['predicted_'+obs_var+' (RF Regression '+str(i_model)+')'] = tmp_cv#[:, i_obs]
                ubrmse = mean_squared_error(df[df.index.year>=val_split_year+1]['predicted_'+obs_var+' (RF Regression '+str(i_model)+')'].dropna() -\
                                    df[df.index.year>=val_split_year+1]['predicted_'+obs_var+' (RF Regression '+str(i_model)+')'].dropna().mean(), 
                                 df[df.index.year>=val_split_year+1]['observed_'+obs_var].dropna() -\
                                    df[df.index.year>=val_split_year+1]['observed_'+obs_var].dropna().mean(), squared=False)
                rho = scipy.stats.pearsonr(df[df.index.year>=val_split_year+1]['predicted_'+obs_var+' (RF Regression '+str(i_model)+')'].dropna(),
                                           df[df.index.year>=val_split_year+1]['observed_'+obs_var].dropna())
                bias = np.mean(df[df.index.year>=val_split_year+1]['predicted_'+obs_var+' (RF Regression '+str(i_model)+')'].dropna() - \
                              df[df.index.year>=val_split_year+1]['observed_'+obs_var].dropna())
                pd_regr_coef[str(i_model)] = regr_cv.feature_importances_#[i_obs,:]
                ubrmse_list.append(ubrmse)
                rho_list.append(rho[0])
                bias_list.append(bias)
                
                i_model += 1
            pd_regr_coef['mean'] = pd_regr_coef.mean(axis = 1)
            pd_regr_coef['std'] = pd_regr_coef.std(axis = 1)
            regr_coef_all[obs_var] = pd_regr_coef
            df_perf['ubrmse_'+obs_var] = ubrmse_list
            df_perf['bias_'+obs_var] = bias_list
            df_perf['rho_'+obs_var] = rho_list
            df_perf.loc['mean'] = df_perf.mean()
            df_perf.loc['std'] = df_perf.std()
    feature_importance = pd_regr_coef.transpose().values
    vmax = np.max(feature_importance)
    vmin = np.min(feature_importance)
    fig, ax = plt.subplots(figsize=(14,14))
    heatmap = sns.heatmap(feature_importance, 
    #                       mask=mask,
                          vmin=-vmax, vmax=vmax, cmap='RdBu_r')
    # Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
    heatmap.set_title('Feature importance of RF regression on different CV fold', fontdict={'fontsize':12}, pad=12);
    # ax.set_xticks([i for _ in range(feature_importance.shape[1])]) 
    ax.set_xticklabels(input_list, fontsize=12, rotation = 90, ha="right")
    ax.set_yticklabels(pd_regr_coef.columns)

    pd_regr_coef.index = input_list
    print(pd_regr_coef['mean'].abs().nlargest(20))
    print(pd_regr_coef['mean'].abs().nlargest(20).index)

    fig, ax = plt.subplots(figsize=(7,7))
    plt.plot(pd_regr_coef['mean'].abs().nlargest(20)/np.sum(pd_regr_coef['mean'].abs()))
    plt.xticks(rotation=90)
    plt.xlabel('absolute value of weight of LSVs / sum of all abs values of all weights of all LSVs')
    
    # calculate NSC for each model
    df = pd.DataFrame()
    X_all = df_input_output[input_list]
    X_all_norm = (X_all - X.mean())/X.std()
    regr_nsc_all = {}
    for i_obs, obs_var in enumerate(output_list):
        i_model = 0
        pd_nsc_shap = pd.DataFrame()
        for i_cv, regr_cv in enumerate(regr_cv_list):
            temp_data = df_input_output[input_list+output_list]
            model = regr_cv
            input_var_list_prac = input_list
            output_var_list = ['observed_'+var for var in output_list]
            output_var_list_pre = ['predicted_'+var for var in output_list]
            obs_var_list = output_list
            j_mat, nsc_mat = jacobian(temp_data, model, input_var_list_prac, output_var_list,
             output_var_list_pre, obs_var_list
             )
            for i_var, var in enumerate(input_list):
                if jacobian_or_nsc == "jacobian":
                    df[input_list[i_var]+'_'+str(i_cv)] = j_mat[var+'_'+obs_var]
                if jacobian_or_nsc == "nsc":
                    df[input_list[i_var]+'_'+str(i_cv)] = nsc_mat[var+'_'+obs_var]
                df.index = df_input_output.index
    df_mean = pd.DataFrame()
    df_std = pd.DataFrame()
    for i_var, var in enumerate(input_list):
        df_tmp = df[[var+'_'+str(i) for i in range(len(regr_cv_list))]]
        df_mean[var] = df_tmp.mean(axis=1)
        df_std[var] = df_tmp.std(axis=1)
    
    # # calculate shap for each model
    # df = pd.DataFrame()
    # X_all = df_input_output[input_list]
    # X_all_norm = (X_all - X.mean())/X.std()
    # regr_shap_all = {}
    # # df['predicted_slope (Lasso Regression opt)'] = lasso_opt.predict(X_all_norm) * y.std()+y.mean()
    # for i_obs, obs_var in enumerate(output_list):
        
    #     # df.index = df_input_output.index
    #     i_model = 0
    #     pd_regr_shap = pd.DataFrame()
    #     for i_cv, regr_cv in enumerate(regr_cv_list):
    #         explainer = shap.TreeExplainer(regr_cv)
    #         shap_values = explainer.shap_values(X_all_norm)
    #         for i_var, var in enumerate(input_list):
    #             df[input_list[i_var]+'_'+str(i_cv)] = shap_values[:, i_var]
    #             df.index = df_input_output.index
    # df_mean = pd.DataFrame()
    # df_std = pd.DataFrame()
    # for i_var, var in enumerate(input_list):
    #     df_tmp = df[[var+'_'+str(i) for i in range(len(regr_cv_list))]]
    #     df_mean[var] = df_tmp.mean(axis=1)
    #     df_std[var] = df_tmp.std(axis=1)
    return regr_cv_list, df_perf, regr_coef_all, df_mean, df_std, df

def svr_cv_train(gpi_data, input_list, output_list, val_split_year,
                 jacobian_or_nsc):
    if len(gpi_data) > 0:
        alphas = np.logspace(-4, -0.5, 30)
        df_input_output = gpi_data[input_list+output_list].copy()
        X = df_input_output[input_list][df_input_output.index.year < val_split_year]
        X_norm = (X - X.mean())/X.std()
        y = df_input_output[output_list][df_input_output.index.year < val_split_year]
        y_norm = (y - y.mean())/y.std()
        
        year_list = df_input_output[df_input_output.index.year < val_split_year].index.year.unique()
        k_fold = KFold(len(year_list), shuffle=True, 
                   random_state=0)
        alpha_opt = 0
        lasso_opt = 0
        k_cv = 0
        score_opt = 10
        svr_lin_list = []
        alpha_list = []
        
        for k, (train, test) in enumerate(k_fold.split(X_norm, y_norm)):
            # svr_lin = SVR(kernel="linear", C=5, gamma="auto")
            svr_lin = SVR(kernel="rbf", C=5, gamma="auto")
            svr_lin.fit(X_norm.values[train], y_norm.values[train])
            score_k = mean_squared_error(
                            svr_lin.predict(X_norm.values[test]) * y.std().values+y.mean().values, 
                            y_norm.values[test] * y.std().values+y.mean().values,
                            squared=False)
            print(
                "[fold {0}] score: {1:.5f}".format(
                    k, 
                    score_k
                )
            )
            print(test)
            svr_lin_list.append(svr_lin)
            if score_k <= score_opt:
                svr_lin_opt = svr_lin
                score_opt = score_k
                k_cv = k
                
        print("Answer: Not very much since we obtained different alphas for different")
        print("subsets of the data and moreover, the scores for these alphas differ")
        print("quite substantially.")
        
        df = pd.DataFrame()
        df_perf = pd.DataFrame()
        X_all = df_input_output[input_list]
        X_all_norm = (X_all - X.mean())/X.std()
        svr_lin_coef_all = {}
        # df['predicted_slope (Lasso Regression opt)'] = lasso_opt.predict(X_all_norm) * y.std()+y.mean()
        for i_obs, obs_var in enumerate(output_list):
            tmp = svr_lin_opt.predict(X_all_norm) * y.std().values+y.mean().values
            df['predicted_'+obs_var+' (Lasso Regression opt)'] = tmp#[:, i_obs]
            df['observed_'+obs_var] = df_input_output[obs_var].values
            df.index = df_input_output.index
            i_model = 0
            bias_list, rho_list, ubrmse_list = [],[],[]
            regr_cv_coef_list = []
            pd_svr_lin_coef = pd.DataFrame()
            for svr_lin in svr_lin_list:
                tmp_cv = svr_lin.predict(X_all_norm) * y.std().values+y.mean().values
                df['predicted_'+obs_var+' (SV Regression '+str(i_model)+')'] = tmp_cv#[:, i_obs]
                ubrmse = mean_squared_error(df[df.index.year>=val_split_year+1]['predicted_'+obs_var+' (SV Regression '+str(i_model)+')'].dropna() -\
                                    df[df.index.year>=val_split_year+1]['predicted_'+obs_var+' (SV Regression '+str(i_model)+')'].dropna().mean(), 
                                 df[df.index.year>=val_split_year+1]['observed_'+obs_var].dropna() -\
                                    df[df.index.year>=val_split_year+1]['observed_'+obs_var].dropna().mean(), squared=False)
                rho = scipy.stats.pearsonr(df[df.index.year>=val_split_year+1]['predicted_'+obs_var+' (SV Regression '+str(i_model)+')'].dropna(),
                                           df[df.index.year>=val_split_year+1]['observed_'+obs_var].dropna())
                bias = np.mean(df[df.index.year>=val_split_year+1]['predicted_'+obs_var+' (SV Regression '+str(i_model)+')'].dropna() - \
                              df[df.index.year>=val_split_year+1]['observed_'+obs_var].dropna())
                # pd_svr_lin_coef[str(i_model)] = svr_lin.coef_[0,:]#[i_obs,:]
                ubrmse_list.append(ubrmse)
                rho_list.append(rho[0])
                bias_list.append(bias)
                
                i_model += 1
            # pd_svr_lin_coef['mean'] = pd_svr_lin_coef.mean(axis = 1)
            # pd_svr_lin_coef['std'] = pd_svr_lin_coef.std(axis = 1)
            # svr_lin_coef_all[obs_var] = pd_svr_lin_coef
            df_perf['ubrmse_'+obs_var] = ubrmse_list
            df_perf['bias_'+obs_var] = bias_list
            df_perf['rho_'+obs_var] = rho_list
            df_perf.loc['mean'] = df_perf.mean()
            df_perf.loc['std'] = df_perf.std()

    # calculate NSC for each model
    df = pd.DataFrame()
    X_all = df_input_output[input_list]
    X_all_norm = (X_all - X.mean())/X.std()
    regr_nsc_all = {}
    for i_obs, obs_var in enumerate(output_list):
        i_model = 0
        pd_nsc_shap = pd.DataFrame()
        for i_cv, svr_lin in enumerate(svr_lin_list):
            temp_data = df_input_output[input_list+output_list]
            model = svr_lin
            input_var_list_prac = input_list
            output_var_list = ['observed_'+var for var in output_list]
            output_var_list_pre = ['predicted_'+var for var in output_list]
            obs_var_list = output_list
            j_mat, nsc_mat = jacobian(temp_data, model, input_var_list_prac, output_var_list,
             output_var_list_pre, obs_var_list
             )
            print(i_cv)
            for i_var, var in enumerate(input_list):
                if jacobian_or_nsc == "jacobian":
                    df[input_list[i_var]+'_'+str(i_cv)] = j_mat[var+'_'+obs_var]
                if jacobian_or_nsc == "nsc":
                    df[input_list[i_var]+'_'+str(i_cv)] = nsc_mat[var+'_'+obs_var]
                df.index = df_input_output.index
    df_mean = pd.DataFrame()
    df_std = pd.DataFrame()
    for i_var, var in enumerate(input_list):
        df_tmp = df[[var+'_'+str(i) for i in range(len(svr_lin_list))]]
        df_mean[var] = df_tmp.mean(axis=1)
        df_std[var] = df_tmp.std(axis=1)
    return svr_lin_list, df_perf, svr_lin_coef_all, df_mean, df_std, df

def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = tf.keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


#%%
if __name__ == "__main__":
    save_path = '/User/xushan/research/TUD/test_jakknife/diff_DNN/'
    with open('{}{}'.format(save_path, '/gpi_data'), 'rb') as f:
        gpi_data = pickle.load(f)
    #%%
    if len(gpi_data) > 0:
        # basic experiment of 16 variables
        input_list = ['WG1', 'WG2', 'WG3', 'WG4', 'WG5', 'WG6', 
                      'WG7', 'WG8', 'WG9', 'WG10',
                      'RN_ISBA', 'GPP_ISBA','WR_ISBA', 'LAI_ISBA',
                      'LAI_ISBA', 'XRS_ISBA']
        output_list = ['sig', 'slop']
        val_split_year = 2017
        lasso_cv_list_basic, df_perf_basic, lasso_coef_all_basic = lasso_cv_train(gpi_data, input_list, 
                                                         output_list, 
                                                         val_split_year)
        


    #%%
    input_list = ['WG1', 'WG2', 'WG3', 'WG4', 'WG5', 'WG6', 
                  'WG7', 'WG8', 'WG9', 'WG10',
                  'RN_ISBA', 'GPP_ISBA','WR_ISBA', 'LAI_ISBA','XRS_ISBA']
    output_list = ['sig'] # , 'slop'
    val_split_year = 2017
    jacobian_or_nsc = "jacobian"
    regr_orignal_sig, df_perf_orignal_sig, regr_coef_all_orignal_sig, df_mean_sig_orignal, df_std_sig_orignal, df_sig_orignal = RandomForestRegressor_cv_train(gpi_data, input_list, 
                                                     output_list, 
                                                     val_split_year, jacobian_or_nsc)
    
    output_list = ['slop'] # , 'slop'
    val_split_year = 2017
    jacobian_or_nsc = "jacobian"
    regr_orignal_slop, df_perf_orignal_slop, regr_coef_all_orignal_slop, df_mean_slop_orignal, df_std_slop_orignal, df_slop_orignal = RandomForestRegressor_cv_train(gpi_data, input_list, 
                                                     output_list, 
                                                     val_split_year, jacobian_or_nsc)
    # input_list = ['WG1', 'WG2', 'WG3', 'WG4', 'WG5', 'WG6', 
    #               'WG7', 'WG8', 'WG9', 'WG10',
    #               'RN_ISBA', 'GPP_ISBA','WR_ISBA', 'LAI_ISBA','XRS_ISBA']
    # output_list = ['sig'] # , 'slop'
    # val_split_year = 2017
    # jacobian_or_nsc = "nsc"
    # regr_orignal_sig, df_perf_orignal_sig, regr_coef_all_orignal_sig, df_mean_sig_orignal_nsc, df_std_sig_orignal_nsc = RandomForestRegressor_cv_train(gpi_data, input_list, 
    #                                                  output_list, 
    #                                                  val_split_year, jacobian_or_nsc)
    
    # output_list = ['slop'] # , 'slop'
    # val_split_year = 2017
    # jacobian_or_nsc = "nsc"
    # regr_orignal_slop, df_perf_orignal_slop, regr_coef_all_orignal_slop, df_mean_slop_orignal_nsc, df_std_slop_orignal_nsc = RandomForestRegressor_cv_train(gpi_data, input_list, 
    #                                                  output_list, 
    #                                                  val_split_year, jacobian_or_nsc)
    
    #%%
    input_list = ['WG3', 'WR_ISBA', 'LAI_ISBA']
    output_list = ['sig'] # , 'slop'
    val_split_year = 2017
    jacobian_or_nsc = "jacobian"
    regr_rzsm_sig, df_perf_rzsm_sig, regr_coef_all_rzsm_sig, df_mean_sig_rzsm, df_std_sig_rzsm, df_sig_rzsm = RandomForestRegressor_cv_train(gpi_data, input_list, 
                                                     output_list, 
                                                     val_split_year, jacobian_or_nsc)
    
    input_list = ['rzsm', 'LAI_ISBA']
    output_list = ['slop'] # , 'slop'
    val_split_year = 2017
    jacobian_or_nsc = "jacobian"
    regr_rzsm_slop, df_perf_rzsm_slop, regr_coef_all_rzsm_slop, df_mean_slop_rzsm, df_std_slop_rzsm, df_slop_rzsm = RandomForestRegressor_cv_train(gpi_data, input_list, 
                                                     output_list, 
                                                     val_split_year, jacobian_or_nsc)
    #%%
    input_list = ['WG3', 'WR_ISBA', 'LAI_ISBA']
    output_list = ['sig'] # , 'slop'
    val_split_year = 2017
    jacobian_or_nsc = "jacobian"
    regr_lessWGs_sig, df_perf_lessWGs_sig, regr_coef_all_lessWGs_sig, df_mean_sig_lessWGs, df_std_sig_lessWGs, df_sig_lessWGs = RandomForestRegressor_cv_train(gpi_data, input_list, 
                                                     output_list, 
                                                     val_split_year, jacobian_or_nsc)
    
    input_list = ['WG4', 'WG5', 'LAI_ISBA']
    output_list = ['slop'] # , 'slop'
    val_split_year = 2017
    jacobian_or_nsc = "jacobian"
    regr_lessWGs_slop, df_perf_lessWGs_slop, regr_coef_all_lessWGs_slop, df_mean_slop_lessWGs, df_std_slop_lessWGs, df_slop_lessWGs = RandomForestRegressor_cv_train(gpi_data, input_list, 
                                                     output_list, 
                                                     val_split_year, jacobian_or_nsc)
    print(regr_coef_all_lessWGs_sig['sig'][['mean','std']])
    print(regr_coef_all_lessWGs_slop['slop'][['mean','std']])
    #%%
    # plot std of J of RF and DNN
    save_path = '/Users/xushan/research/TUD/test_jackknife/diff_DNN/fig_L2Dropout_separate/'
    with open('{}{}'.format(save_path, '/df_nsc'), 'rb') as f:
        clusters = pickle.load(f)
    dict_nsc_exp = clusters
    
    ML_list = [
               "/Users/xushan/research/TUD/test_jackknife/diff_DNN/"+\
               "/var_multiSM_lessWGs_L2Dropout/results/", 
               "/Users/xushan/research/TUD/test_jackknife/diff_DNN/"+\
               "/var_multiSM_L2Dropout_separate/sig/results/", 
               "/Users/xushan/research/TUD/test_jackknife/diff_DNN/"+\
               "/var_multiSM_L2Dropout_separate/slop/results/", 
               ]
    exp = '4169'
    exp_list = [
                # "4152",
                "4169",
                # "4188","4191"
                ]
    var_nsc_list = list(dict_nsc_exp[exp][ML_list[1]][1].iloc[0]['nsc'].columns) +\
        list(dict_nsc_exp[exp][ML_list[2]][1].iloc[0]['nsc'].columns)
    df_std_nsc_exp = {}
    df_mean_nsc_exp = {}
    for exp in exp_list:
        df_std_nsc_obs = {}
        df_mean_nsc_obs = {}
        for var_obs in var_nsc_list:
            # df_all_list = df_all_dict[CV][var_obs]
            df_std_nsc_ml = {}
            df_mean_nsc_ml = {}
            for ml_path in ML_list:
                if var_obs in dict_nsc_exp[exp][ml_path][0].iloc[0]['nsc'].columns:
                    df_all_list = dict_nsc_exp[exp][ml_path]
                    df_list = []
                    for i, df_all in enumerate(df_all_list):
                        testdataframe = df_all.iloc[0]['jacob'][var_obs]
                        df_list.append(testdataframe)
                    df_std = pd.concat(df_list, axis=1).std(axis = 1)
                    df_mean = pd.concat(df_list, axis=1).mean(axis = 1)
                    df_std_nsc_ml[ml_path] = df_std
                    df_mean_nsc_ml[ml_path] = df_mean
            df_std_nsc_obs[var_obs] = df_std_nsc_ml
            df_mean_nsc_obs[var_obs] = df_mean_nsc_ml
        df_std_nsc_exp[exp] = df_std_nsc_obs
        df_mean_nsc_exp[exp] = df_mean_nsc_obs
        
    
    #%%
    # plot figure for NSCs in random forest
    # mean and std
    # 3 experiment, orignal, rzsm, lessWGs
    # [sig, WG3], [sig, LAI_ISBA], [slop, LAI], [slop, WG3], [slop,rzsm]
    exp_name = ['orignal', 'rzsm', 'lessWGs']
    var_exp_dict_sig = {'orignal':['WG3', 'LAI_ISBA'],
                        'rzsm':['WG3', 'LAI_ISBA'],
                        'lessWGs':['WG3', 'LAI_ISBA']}
    var_exp_dict_slop = {'orignal':['WG4', 'WG5', 'LAI_ISBA'],
                            'rzsm':['rzsm', 'LAI_ISBA'],
                            'lessWGs':['WG4', 'LAI_ISBA']}
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_mean_sig_orignal['WG3'].groupby(df_mean_sig_orignal.index.dayofyear).mean(),
            color='red',label='regr_original')
    ax.plot(df_mean_sig_rzsm['WG3'].groupby(df_mean_sig_rzsm.index.dayofyear).mean(),
            color='blue',label='regr_rzsm')
    ax.plot(df_mean_sig_lessWGs['WG3'].groupby(df_mean_sig_lessWGs.index.dayofyear).mean(),
            color='green',label='regr_lessWGs')
    df_ml = df_mean_nsc_exp[exp]['Input_WG3_sig']['/Users/xushan/research/TUD/test_jackknife/diff_DNN//var_multiSM_L2Dropout_separate/sig/results/']
    df_ml = df_ml[df_ml.index.year >= 2017]
    ax.plot(df_ml.groupby(df_ml.index.dayofyear).mean(),
            color='purple',label='DNN_lessWGs_separate')
    ax.set_ylabel('J(sig40, WG3)')
    plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)
    
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_mean_sig_orignal['LAI_ISBA'].groupby(df_mean_sig_orignal.index.dayofyear).mean(),
            color='red',label='regr_original')
    ax.plot(df_mean_sig_rzsm['LAI_ISBA'].groupby(df_mean_sig_rzsm.index.dayofyear).mean(),
            color='blue',label='regr_rzsm')
    ax.plot(df_mean_sig_lessWGs['LAI_ISBA'].groupby(df_mean_sig_lessWGs.index.dayofyear).mean(),
            color='green',label='regr_lessWGs')
    df_ml = df_mean_nsc_exp[exp]['Input_LAI_ISBA_sig']['/Users/xushan/research/TUD/test_jackknife/diff_DNN//var_multiSM_L2Dropout_separate/sig/results/']
    df_ml = df_ml[df_ml.index.year >= 2017]
    ax.plot(df_ml.groupby(df_ml.index.dayofyear).mean(),
            color='purple',label='DNN_lessWGs_separate')
    ax.set_ylabel('J(sig40, LAI)')
    plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)
    
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_mean_slop_orignal['LAI_ISBA'].groupby(df_mean_slop_orignal.index.dayofyear).mean(),
            color='red',label='regr_original')
    ax.plot(df_mean_slop_rzsm['LAI_ISBA'].groupby(df_mean_slop_rzsm.index.dayofyear).mean(),
            color='blue',label='regr_rzsm')
    ax.plot(df_mean_slop_lessWGs['LAI_ISBA'].groupby(df_mean_slop_lessWGs.index.dayofyear).mean(),
            color='green',label='regr_lessWGs')
    df_ml = df_mean_nsc_exp[exp]['Input_LAI_ISBA_slop']['/Users/xushan/research/TUD/test_jackknife/diff_DNN//var_multiSM_L2Dropout_separate/slop/results/']
    df_ml = df_ml[df_ml.index.year >= 2017]
    ax.plot(df_ml.groupby(df_ml.index.dayofyear).mean(),
            color='purple',label='DNN_lessWGs_separate')
    ax.set_ylabel('J(slope, LAI)')
    plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)
    
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_mean_slop_lessWGs['WG4'].groupby(df_mean_slop_lessWGs.index.dayofyear).mean(),
            color='red',label='WG4')
    ax.plot(df_mean_slop_lessWGs['WG5'].groupby(df_mean_slop_lessWGs.index.dayofyear).mean(),
            color='green',label='WG5')
    ax.set_ylabel('J(slope, LSV) of regr_lessWGs')
    plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)
    
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_mean_slop_orignal['WG4'].groupby(df_mean_slop_orignal.index.dayofyear).mean(),
            color='red',label='WG4')
    ax.plot(df_mean_slop_orignal['WG5'].groupby(df_mean_slop_orignal.index.dayofyear).mean(),
            color='green',label='WG5')
    ax.plot(df_mean_slop_orignal['WG6'].groupby(df_mean_slop_orignal.index.dayofyear).mean(),
            color='blue',label='WG6')
    ax.plot(df_mean_slop_orignal['WG7'].groupby(df_mean_slop_orignal.index.dayofyear).mean(),
            color='black',label='WG7')
    ax.set_ylabel('J(slope, LSV) of regr_orignal')
    plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)
    
    #%% std of the nscs
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_std_sig_orignal[df_std_sig_orignal.index.year>=2017]['WG3'].groupby(df_std_sig_orignal[df_std_sig_orignal.index.year>=2017].index.dayofyear).mean(),
            color='red',label='regr_original')
    ax.plot(df_std_sig_rzsm[df_std_sig_rzsm.index.year>=2017]['WG3'].groupby(df_std_sig_rzsm[df_std_sig_rzsm.index.year>=2017].index.dayofyear).mean(),
            color='blue',label='regr_rzsm')
    ax.plot(df_std_sig_lessWGs[df_std_sig_lessWGs.index.year>=2017]['WG3'].groupby(df_std_sig_lessWGs[df_std_sig_lessWGs.index.year>=2017].index.dayofyear).mean(),
            color='green',label='regr_lessWGs')
    df_ml = df_std_nsc_exp[exp]['Input_WG3_sig']['/Users/xushan/research/TUD/test_jackknife/diff_DNN//var_multiSM_L2Dropout_separate/sig/results/']
    df_ml = df_ml[df_ml.index.year >= 2017]
    ax.plot(df_ml.groupby(df_ml.index.dayofyear).mean(),
            color='purple',label='DNN_lessWGs_separate')
    ax.set_ylabel('std of J(sig40, WG3)')
    plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)
    
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_std_sig_orignal[df_std_sig_orignal.index.year>=2017]['LAI_ISBA'].groupby(df_std_sig_orignal[df_std_sig_orignal.index.year>=2017].index.dayofyear).mean(),
            color='red',label='regr_original')
    ax.plot(df_std_sig_rzsm[df_std_sig_rzsm.index.year>=2017]['LAI_ISBA'].groupby(df_std_sig_rzsm[df_std_sig_rzsm.index.year>=2017].index.dayofyear).mean(),
            color='blue',label='regr_rzsm')
    ax.plot(df_std_sig_lessWGs[df_std_sig_lessWGs.index.year>=2017]['LAI_ISBA'].groupby(df_std_sig_lessWGs[df_std_sig_lessWGs.index.year>=2017].index.dayofyear).mean(),
            color='green',label='regr_lessWGs')
    df_ml = df_std_nsc_exp[exp]['Input_LAI_ISBA_sig']['/Users/xushan/research/TUD/test_jackknife/diff_DNN//var_multiSM_L2Dropout_separate/sig/results/']
    df_ml = df_ml[df_ml.index.year >= 2017]
    ax.plot(df_ml.groupby(df_ml.index.dayofyear).mean(),
            color='purple',label='DNN_lessWGs_separate')
    ax.set_ylabel('std of J(sig40, LAI)')
    plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)
    
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_std_slop_orignal[df_std_sig_orignal.index.year>=2017]['LAI_ISBA'].groupby(df_std_slop_orignal[df_std_sig_orignal.index.year>=2017].index.dayofyear).mean(),
            color='red',label='regr_original')
    ax.plot(df_std_slop_rzsm[df_std_sig_rzsm.index.year>=2017]['LAI_ISBA'].groupby(df_std_slop_rzsm[df_std_sig_rzsm.index.year>=2017].index.dayofyear).mean(),
            color='blue',label='regr_rzsm')
    ax.plot(df_std_slop_lessWGs[df_std_sig_lessWGs.index.year>=2017]['LAI_ISBA'].groupby(df_std_slop_lessWGs[df_std_sig_lessWGs.index.year>=2017].index.dayofyear).mean(),
            color='green',label='regr_lessWGs')
    df_ml = df_std_nsc_exp[exp]['Input_LAI_ISBA_slop']['/Users/xushan/research/TUD/test_jackknife/diff_DNN//var_multiSM_L2Dropout_separate/slop/results/']
    df_ml = df_ml[df_ml.index.year >= 2017]
    ax.plot(df_ml.groupby(df_ml.index.dayofyear).mean(),
            color='purple',label='DNN_lessWGs_separate')
    ax.set_ylabel('std of J(slope, LAI)')
    plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)
    #%%
    fig, ax = plt.subplots(figsize=(10,4))
    lsv = 'LAI_ISBA'
    color_list = cm.Greens_r(np.linspace(0, 1, 10+2))
    for i in range(10):
        df_ml = df_slop_lessWGs[lsv+'_'+str(i)]
        df_ml = df_ml[df_ml.index.year >= 2017]
        ax.plot(df_ml.groupby(df_ml.index.dayofyear).mean(),
                color=color_list[i],label='regr_lessWGs')
    ax.set_ylabel('J(slope, LAI)')
    plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)
    
    fig, ax = plt.subplots(figsize=(10,4))
    lsv = 'LAI_ISBA'
    color_list = cm.Greens_r(np.linspace(0, 1, 10+2))
    for i in range(10):
        df_ml = dict_nsc_exp['4169']['/Users/xushan/research/TUD/test_jackknife/diff_DNN//var_multiSM_L2Dropout_separate/slop/results/'][i].iloc[0]['jacob']['Input_LAI_ISBA_slop']
        df_ml = df_ml[df_ml.index.year >= 2017]
        ax.plot(df_ml.groupby(df_ml.index.dayofyear).mean(),
                color=color_list[i],label='DNN_L2Dropout_separate')
    ax.set_ylabel('J(slope, LAI)')
    plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)
    #%%
    # plot figure for predictions in random forest
    # mean and std
    # 3 experiment, orignal, rzsm, lessWGs
    # [sig, WG3], [sig, LAI_ISBA], [slop, LAI], [slop, WG3], [slop,rzsm]
    exp_name = ['orignal', 'rzsm', 'lessWGs']
    var_exp_dict_sig = {'orignal':['WG3', 'LAI_ISBA'],
                        'rzsm':['WG3', 'LAI_ISBA'],
                        'lessWGs':['WG3', 'LAI_ISBA']}
    var_exp_dict_slop = {'orignal':['WG4', 'WG5', 'LAI_ISBA'],
                            'rzsm':['rzsm', 'LAI_ISBA'],
                            'lessWGs':['WG4', 'LAI_ISBA']}
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_mean_sig_orignal['WG3'].groupby(df_mean_sig_orignal.index.dayofyear).mean(),
            color='red',label='regr_original')
    ax.plot(df_mean_sig_rzsm['WG3'].groupby(df_mean_sig_rzsm.index.dayofyear).mean(),
            color='blue',label='regr_rzsm')
    ax.plot(df_mean_sig_lessWGs['WG3'].groupby(df_mean_sig_lessWGs.index.dayofyear).mean(),
            color='green',label='regr_lessWGs')
    df_ml = df_mean_nsc_exp[exp]['Input_WG3_sig']['/Users/xushan/research/TUD/test_jackknife/diff_DNN//var_multiSM_L2Dropout_separate/sig/results/']
    df_ml = df_ml[df_ml.index.year >= 2017]
    ax.plot(df_ml.groupby(df_ml.index.dayofyear).mean(),
            color='purple',label='DNN_lessWGs_separate')
    ax.set_ylabel('NSC(sig40, WG3)')
    plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)
    
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_mean_sig_orignal['LAI_ISBA'].groupby(df_mean_sig_orignal.index.dayofyear).mean(),
            color='red',label='regr_original')
    ax.plot(df_mean_sig_rzsm['LAI_ISBA'].groupby(df_mean_sig_rzsm.index.dayofyear).mean(),
            color='blue',label='regr_rzsm')
    ax.plot(df_mean_sig_lessWGs['LAI_ISBA'].groupby(df_mean_sig_lessWGs.index.dayofyear).mean(),
            color='green',label='regr_lessWGs')
    df_ml = df_mean_nsc_exp[exp]['Input_LAI_ISBA_sig']['/Users/xushan/research/TUD/test_jackknife/diff_DNN//var_multiSM_L2Dropout_separate/sig/results/']
    df_ml = df_ml[df_ml.index.year >= 2017]
    ax.plot(df_ml.groupby(df_ml.index.dayofyear).mean(),
            color='purple',label='DNN_lessWGs_separate')
    ax.set_ylabel('NSC(sig40, LAI)')
    plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)
    
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_mean_slop_orignal['LAI_ISBA'].groupby(df_mean_slop_orignal.index.dayofyear).mean(),
            color='red',label='regr_original')
    ax.plot(df_mean_slop_rzsm['LAI_ISBA'].groupby(df_mean_slop_rzsm.index.dayofyear).mean(),
            color='blue',label='regr_rzsm')
    ax.plot(df_mean_slop_lessWGs['LAI_ISBA'].groupby(df_mean_slop_lessWGs.index.dayofyear).mean(),
            color='green',label='regr_lessWGs')
    df_ml = df_mean_nsc_exp[exp]['Input_LAI_ISBA_slop']['/Users/xushan/research/TUD/test_jackknife/diff_DNN//var_multiSM_L2Dropout_separate/slop/results/']
    df_ml = df_ml[df_ml.index.year >= 2017]
    ax.plot(df_ml.groupby(df_ml.index.dayofyear).mean(),
            color='purple',label='DNN_lessWGs_separate')
    ax.set_ylabel('NSC(slope, LAI)')
    plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)
    
    #%%
    input_list = ['WG3', 'WR_ISBA', 'LAI_ISBA']
    output_list = ['sig'] # , 'slop'
    val_split_year = 2017
    regr_gpp_sig, df_perf_gpp_sig, regr_coef_all_gpp_sig = RandomForestRegressor_cv_train(gpi_data, input_list, 
                                                     output_list, 
                                                     val_split_year)
    
    input_list = ['rzsm', 'GPP_ISBA', 'LAI_ISBA']
    output_list = ['slop'] # , 'slop'
    val_split_year = 2017
    regr_gpp_slop, df_perf_gpp_slop, regr_coef_all_gpp_slop = RandomForestRegressor_cv_train(gpi_data, input_list, 
                                                     output_list, 
                                                     val_split_year)
    print(regr_coef_all_gpp_sig['sig'][['mean','std']])
    print(regr_coef_all_gpp_slop['slop'][['mean','std']])
    #%%
    input_list = ['WG1', 'WG2', 'WG3', 'WG4', 'WG5', 'WG6', 
                  'WG7', 'WG8', 'WG9', 'WG10',
                  'RN_ISBA', 'GPP_ISBA','WR_ISBA', 'LAI_ISBA','XRS_ISBA']
    output_list = ['sig'] # , 'slop'
    val_split_year = 2017
    jacobian_or_nsc = "jacobian"
    svr_original_sig, df_perf_original_sig, svr_coef_all_original_sig, df_mean_sig_orignal, df_std_sig_orignal, df_sig_orignal = svr_cv_train(gpi_data, input_list, 
                                                     output_list, 
                                                     val_split_year, jacobian_or_nsc)
    
    input_list = ['WG1', 'WG2', 'WG3', 'WG4', 'WG5', 'WG6', 
                  'WG7', 'WG8', 'WG9', 'WG10',
                  'RN_ISBA', 'GPP_ISBA','WR_ISBA', 'LAI_ISBA','XRS_ISBA']
    output_list = ['slop'] # , 'slop'
    val_split_year = 2017
    jacobian_or_nsc = "jacobian"
    svr_original_slop, df_perf_original_slop, svr_coef_all_original_slop, df_mean_slop_orignal, df_std_slop_orignal, df_slop_orignal = svr_cv_train(gpi_data, input_list, 
                                                     output_list, 
                                                     val_split_year, jacobian_or_nsc)
    #%%
    input_list = ['WG3', 'WR_ISBA', 'LAI_ISBA']
    output_list = ['sig'] # , 'slop'
    val_split_year = 2017
    jacobian_or_nsc = "jacobian"
    svr_rzsm_sig, df_perf_rzsm_sig, svr_coef_all_rzsm_sig, df_mean_sig_rzsm, df_std_sig_rzsm, df_sig_rzsm = svr_cv_train(gpi_data, input_list, 
                                                     output_list, 
                                                     val_split_year, jacobian_or_nsc)
    
    input_list = ['rzsm', 'LAI_ISBA']
    output_list = ['slop'] # , 'slop'
    val_split_year = 2017
    jacobian_or_nsc = "jacobian"
    svr_rzsm_slop, df_perf_rzsm_slop, svr_coef_all_rzsm_slop, df_mean_slop_rzsm, df_std_slop_rzsm, df_slop_rzsm = svr_cv_train(gpi_data, input_list, 
                                                     output_list, 
                                                     val_split_year, jacobian_or_nsc)
    #%%
    input_list = ['WG3', 'WR_ISBA', 'LAI_ISBA']
    output_list = ['sig'] # , 'slop'
    val_split_year = 2017
    jacobian_or_nsc = "jacobian"
    svr_lessWGs_sig, df_perf_lessWGs_sig, svr_coef_all_lessWGs_sig, df_mean_sig_lessWGs, df_std_sig_lessWGs, df_sig_lessWGs = svr_cv_train(gpi_data, input_list, 
                                                     output_list, 
                                                     val_split_year, jacobian_or_nsc)
    
    input_list = ['WG4', 'WG5', 'LAI_ISBA']
    output_list = ['slop'] # , 'slop'
    val_split_year = 2017
    jacobian_or_nsc = "jacobian"
    svr_lessWGs_slop, df_perf_lessWGs_slop, svr_coef_all_lessWGs_slop, df_mean_slop_lessWGs, df_std_slop_lessWGs, df_slop_lessWGs = svr_cv_train(gpi_data, input_list, 
                                                     output_list, 
                                                     val_split_year, jacobian_or_nsc)
    #%%
    # plot figure for NSCs in SVR
    # mean and std
    # 3 experiment, orignal, rzsm, lessWGs
    # [sig, WG3], [sig, LAI_ISBA], [slop, LAI], [slop, WG3], [slop,rzsm]
    exp_name = ['orignal', 'rzsm', 'lessWGs']
    var_exp_dict_sig = {'orignal':['WG3', 'LAI_ISBA'],
                        'rzsm':['WG3', 'LAI_ISBA'],
                        'lessWGs':['WG3', 'LAI_ISBA']}
    var_exp_dict_slop = {'orignal':['WG4', 'WG5', 'LAI_ISBA'],
                            'rzsm':['rzsm', 'LAI_ISBA'],
                            'lessWGs':['WG4', 'LAI_ISBA']}
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_mean_sig_orignal['WG3'].groupby(df_mean_sig_orignal.index.dayofyear).mean(),
            color='red',label='regr_original')
    ax.plot(df_mean_sig_rzsm['WG3'].groupby(df_mean_sig_rzsm.index.dayofyear).mean(),
            color='blue',label='regr_rzsm')
    ax.plot(df_mean_sig_lessWGs['WG3'].groupby(df_mean_sig_lessWGs.index.dayofyear).mean(),
            color='green',label='regr_lessWGs')
    ax.set_ylabel('J(sig40, WG3)')
    plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)
    
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_mean_sig_orignal['LAI_ISBA'].groupby(df_mean_sig_orignal.index.dayofyear).mean(),
            color='red',label='regr_original')
    ax.plot(df_mean_sig_rzsm['LAI_ISBA'].groupby(df_mean_sig_rzsm.index.dayofyear).mean(),
            color='blue',label='regr_rzsm')
    ax.plot(df_mean_sig_lessWGs['LAI_ISBA'].groupby(df_mean_sig_lessWGs.index.dayofyear).mean(),
            color='green',label='regr_lessWGs')
    ax.set_ylabel('J(sig40, LAI)')
    plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)
    
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_mean_slop_orignal['LAI_ISBA'].groupby(df_mean_slop_orignal.index.dayofyear).mean(),
            color='red',label='regr_original')
    ax.plot(df_mean_slop_rzsm['LAI_ISBA'].groupby(df_mean_slop_rzsm.index.dayofyear).mean(),
            color='blue',label='regr_rzsm')
    ax.plot(df_mean_slop_lessWGs['LAI_ISBA'].groupby(df_mean_slop_lessWGs.index.dayofyear).mean(),
            color='green',label='regr_lessWGs')
    ax.set_ylabel('J(slope, LAI)')
    plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)
    
    #%% std of the nscs
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_std_sig_orignal['WG3'].groupby(df_std_sig_orignal.index.dayofyear).mean(),
            color='red',label='regr_original')
    ax.plot(df_std_sig_rzsm['WG3'].groupby(df_std_sig_rzsm.index.dayofyear).mean(),
            color='blue',label='regr_rzsm')
    ax.plot(df_std_sig_lessWGs['WG3'].groupby(df_std_sig_lessWGs.index.dayofyear).mean(),
            color='green',label='regr_lessWGs')
    ax.set_ylabel('std of J(sig40, WG3)')
    plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)
    
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_std_sig_orignal['LAI_ISBA'].groupby(df_std_sig_orignal.index.dayofyear).mean(),
            color='red',label='regr_original')
    ax.plot(df_std_sig_rzsm['LAI_ISBA'].groupby(df_std_sig_rzsm.index.dayofyear).mean(),
            color='blue',label='regr_rzsm')
    ax.plot(df_std_sig_lessWGs['LAI_ISBA'].groupby(df_std_sig_lessWGs.index.dayofyear).mean(),
            color='green',label='regr_lessWGs')
    ax.set_ylabel('std of J(sig40, LAI)')
    plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)
    
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_std_slop_orignal['LAI_ISBA'].groupby(df_std_slop_orignal.index.dayofyear).mean(),
            color='red',label='regr_original')
    ax.plot(df_std_slop_rzsm['LAI_ISBA'].groupby(df_std_slop_rzsm.index.dayofyear).mean(),
            color='blue',label='regr_rzsm')
    ax.plot(df_std_slop_lessWGs['LAI_ISBA'].groupby(df_std_slop_lessWGs.index.dayofyear).mean(),
            color='green',label='regr_lessWGs')
    ax.set_ylabel('std of J(slope, LAI)')
    plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)
    
    #%% compare RF, SVR and DNN models in one plot
    # expriment: 
    # regr_lessWGs_slop
    # svr_lessWGs_slop
    # no DNN yet!
    input_list = ['WG3', 'WR_ISBA', 'LAI_ISBA']
    output_list = ['sig'] # , 'slop'
    val_split_year = 2017
    if len(gpi_data) > 0:
        alphas = np.logspace(-4, -0.5, 30)
        df_input_output = gpi_data[input_list+output_list].copy()
        X = df_input_output[input_list][df_input_output.index.year < val_split_year]
        X_norm = (X - X.mean())/X.std()
        y = df_input_output[output_list][df_input_output.index.year < val_split_year]
        y_norm = (y - y.mean())/y.std()
        
        X_all = df_input_output[input_list]
        X_all_norm = (X_all - X.mean())/X.std()

        
        i_model = 0
        df = pd.DataFrame()
        df['observed_'+output_list[0]] = df_input_output[output_list]
        
        regr_cv_list = regr_lessWGs_sig
        for regr_cv in regr_cv_list:
            df['predicted_sig (RF Regression '+str(i_model)+')'] = regr_cv.predict(X_all_norm) * y.std().values+y.mean().values
            df_plot = df#pd.concat([df, df_time], axis=1)
            i_model += 1
        i_model = 0
        regr_cv_list = svr_lessWGs_sig
        for regr_cv in regr_cv_list:
            df['predicted_sig (SVR Regression '+str(i_model)+')'] = regr_cv.predict(X_all_norm) * y.std().values+y.mean().values
            df_plot = df#pd.concat([df, df_time], axis=1)
            i_model += 1
        
        i=0
        fig, ax = plt.subplots(figsize=(20,4))
        colors = ['r']+list(cm.Greens_r(np.linspace(0, 1, len(regr_cv_list)))[::-1]) + \
                list(cm.Blues_r(np.linspace(0, 1, len(regr_cv_list)))[::-1])
        style_list = ['-']*len(regr_lessWGs_sig + svr_lessWGs_sig)
        linewidths = [1]*len(regr_lessWGs_sig + svr_lessWGs_sig)
        label_list = df_plot.columns#[0:-1]
    
        for col, style,color, lw,label in zip(df_plot.columns, style_list,
                                        colors, linewidths,label_list):
            df_plot[df_plot.index.year>=val_split_year][col].plot(color=color, style=style, markersize=2,lw=lw, ax=ax,label=label).legend(loc='upper right')
            i+=1
        ax.set_ylabel(output_list[0], fontsize=13)
        plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)

    input_list = ['WG4', 'WG5', 'LAI_ISBA']
    output_list = ['slop']
    val_split_year = 2017
    if len(gpi_data) > 0:
        alphas = np.logspace(-4, -0.5, 30)
        df_input_output = gpi_data[input_list+output_list].copy()
        X = df_input_output[input_list][df_input_output.index.year < val_split_year]
        X_norm = (X - X.mean())/X.std()
        y = df_input_output[output_list][df_input_output.index.year < val_split_year]
        y_norm = (y - y.mean())/y.std()
        
        X_all = df_input_output[input_list]
        X_all_norm = (X_all - X.mean())/X.std()

        
        i_model = 0
        df = pd.DataFrame()
        df['observed_'+output_list[0]] = df_input_output[output_list]
        
        regr_cv_list = regr_lessWGs_slop
        for regr_cv in regr_cv_list:
            df['predicted_slope (RF Regression '+str(i_model)+')'] = regr_cv.predict(X_all_norm) * y.std().values+y.mean().values
            df_plot = df#pd.concat([df, df_time], axis=1)
            i_model += 1
        i_model = 0
        regr_cv_list = svr_lessWGs_slop
        for regr_cv in regr_cv_list:
            df['predicted_slope (SVR Regression '+str(i_model)+')'] = regr_cv.predict(X_all_norm) * y.std().values+y.mean().values
            df_plot = df#pd.concat([df, df_time], axis=1)
            i_model += 1
        
        i=0
        fig, ax = plt.subplots(figsize=(20,4))
        colors = ['r']+list(cm.Greens_r(np.linspace(0, 1, len(regr_cv_list)))[::-1]) + \
                list(cm.Blues_r(np.linspace(0, 1, len(regr_cv_list)))[::-1])
        style_list = ['-']*len(regr_lessWGs_slop + svr_lessWGs_slop)
        linewidths = [1]*len(regr_lessWGs_slop + svr_lessWGs_slop)
        label_list = df_plot.columns#[0:-1]
    
        for col, style,color, lw,label in zip(df_plot.columns, style_list,
                                        colors, linewidths,label_list):
            df_plot[df_plot.index.year>=val_split_year][col].plot(color=color, style=style, markersize=2,lw=lw, ax=ax,label=label).legend(loc='upper right')
            i+=1
        ax.set_ylabel(output_list[0], fontsize=13)
        plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)
    #%% compare std of RF, SVR and DNN model predictions in one plot
    # expriment: 
    # regr_lessWGs_slop
    # svr_lessWGs_slop
    # no DNN yet!
    input_list = ['WG3', 'WR_ISBA', 'LAI_ISBA']
    output_list = ['sig'] # , 'slop'
    val_split_year = 2017
    if len(gpi_data) > 0:
        alphas = np.logspace(-4, -0.5, 30)
        df_input_output = gpi_data[input_list+output_list].copy()
        X = df_input_output[input_list][df_input_output.index.year < val_split_year]
        X_norm = (X - X.mean())/X.std()
        y = df_input_output[output_list][df_input_output.index.year < val_split_year]
        y_norm = (y - y.mean())/y.std()
        
        X_all = df_input_output[input_list]
        X_all_norm = (X_all - X.mean())/X.std()

        
        i_model = 0
        df_std = pd.DataFrame()
        df = pd.DataFrame()
        # df['observed_'+output_list[0]] = df_input_output[output_list]
        
        regr_cv_list = regr_lessWGs_sig
        for regr_cv in regr_cv_list:
            df['predicted_sig (RF Regression '+str(i_model)+')'] = regr_cv.predict(X_all_norm) * y.std().values+y.mean().values
            df_plot = df#pd.concat([df, df_time], axis=1)
            i_model += 1
        df_std['RF'] = df.std(axis = 1)
        df = pd.DataFrame()
        i_model = 0
        regr_cv_list = svr_lessWGs_sig
        for regr_cv in regr_cv_list:
            df['predicted_sig (SVR Regression '+str(i_model)+')'] = regr_cv.predict(X_all_norm) * y.std().values+y.mean().values
            df_plot = df#pd.concat([df, df_time], axis=1)
            i_model += 1
        df_std['SVR'] = df.std(axis = 1)
        i=0
        fig, ax = plt.subplots(figsize=(20,4))
        df_std['RF'].plot(ax=ax,color='green', label='Regr_lessWGs')
        df_std['SVR'].plot(ax=ax,color='red', label='Svm_lin_lessWGs')
        ax.set_ylabel(output_list[0], fontsize=13)
        plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)

    input_list = ['WG4', 'WG5', 'LAI_ISBA']
    output_list = ['slop']
    val_split_year = 2017
    if len(gpi_data) > 0:
        alphas = np.logspace(-4, -0.5, 30)
        df_input_output = gpi_data[input_list+output_list].copy()
        X = df_input_output[input_list][df_input_output.index.year < val_split_year]
        X_norm = (X - X.mean())/X.std()
        y = df_input_output[output_list][df_input_output.index.year < val_split_year]
        y_norm = (y - y.mean())/y.std()
        
        X_all = df_input_output[input_list]
        X_all_norm = (X_all - X.mean())/X.std()

        
        i_model = 0
        df = pd.DataFrame()
        # df['observed_'+output_list[0]] = df_input_output[output_list]
        
        regr_cv_list = regr_lessWGs_slop
        for regr_cv in regr_cv_list:
            df['predicted_slope (RF Regression '+str(i_model)+')'] = regr_cv.predict(X_all_norm) * y.std().values+y.mean().values
            df_plot = df#pd.concat([df, df_time], axis=1)
            i_model += 1
        df_std['RF'] = df.std(axis = 1)
        df = pd.DataFrame()

        i_model = 0
        regr_cv_list = svr_lessWGs_slop
        for regr_cv in regr_cv_list:
            df['predicted_slope (SVR Regression '+str(i_model)+')'] = regr_cv.predict(X_all_norm) * y.std().values+y.mean().values
            df_plot = df#pd.concat([df, df_time], axis=1)
            i_model += 1
        df_std['SVR'] = df.std(axis = 1)
        i=0
        fig, ax = plt.subplots(figsize=(20,4))
        df_std['RF'].plot(ax=ax,color='green', label='Regr_lessWGs')
        df_std['SVR'].plot(ax=ax,color='red', label='Svm_lin_lessWGs')
        ax.set_ylabel(output_list[0], fontsize=13)
        plt.legend(bbox_to_anchor=(0.35,-0.3,0.4,0.2), ncol=2, fontsize=13)