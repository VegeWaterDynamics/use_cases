
#In this code, last 100 rows of each data hold out in order to use it as a test data because if we first stack the data and split the data
# we don't know which sample and which time chose so we can't reshape the y_eval and average over fields for each time.
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.metrics import mean_squared_error , r2_score,  mean_absolute_error
from scipy.stats import pearsonr,spearmanr
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
import datetime
import matplotlib.dates as mdates

from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU

#Loading DSSAT outputs
#___________________________________________________________________________________________________________________________________________

year="2018_n"
YY = 2018
# Leaf area indexx
Brabant_LAI=pd.read_pickle(f"/home/tnikaein/Documents/PhD/PythonProjects/Netherland_barabant/{(year)}/Results/brabant_LAI.pkl")
Brabant_LAI.index = Brabant_LAI.index.date

# first layer soil water
Brabant_SWTD=pd.read_pickle(f"/home/tnikaein/Documents/PhD/PythonProjects/Netherland_barabant/{(year)}/Results/brabant_SWTD.pkl")
Brabant_SWTD.index = Brabant_SWTD.index.date


# Root layer soil water
Brabant_SWTD6 = pd.read_pickle(f"/home/tnikaein/Documents/PhD/PythonProjects/Netherland_barabant/{(year)}/Results/brabant_SWTD6.pkl")
Brabant_SWTD6.index = Brabant_SWTD6.index.date


# Canopy height
Brabant_CHTD = pd.read_pickle(f"/home/tnikaein/Documents/PhD/PythonProjects/Netherland_barabant/{(year)}/Results/brabant_CHTD.pkl")
Brabant_CHTD.index = Brabant_CHTD.index.date

# Top weight
Brabant_CWAD = pd.read_pickle(f"/home/tnikaein/Documents/PhD/PythonProjects/Netherland_barabant/{(year)}/Results/brabant_CWAD.pkl")
Brabant_CWAD.index = Brabant_CWAD.index.date

# # Leaf weight
# Brabant_LWAD = pd.read_pickle(f"/home/tnikaein/Documents/PhD/PythonProjects/Netherland_barabant/{(year)}/Results/brabant_LWAD.pkl")
# Brabant_LWAD.index = Brabant_LWAD.index.date
#
# # Stem weight
# Brabant_SWAD = pd.read_pickle(f"/home/tnikaein/Documents/PhD/PythonProjects/Netherland_barabant/{(year)}/Results/brabant_SWAD.pkl")
# Brabant_SWAD.index = Brabant_SWAD.index.date
#
# # GRAIN weight
# Brabant_GWAD = pd.read_pickle("/home/tnikaein/Documents/PhD/PythonProjects/Netherland/Outputs/DSSAT_code_output/Brabant_GWAD.pkl")
# Brabant_GWAD.index = Brabant_GWAD.index.date
#
# # ROOT DENSITY weight
# Brabant_RL1D = pd.read_pickle("/home/tnikaein/Documents/PhD/PythonProjects/Netherland/Outputs/DSSAT_code_output/Brabant_RL1D.pkl")
# Brabant_RL1D.index = Brabant_RL1D.index.date

# # Rain
# Rain=pd.read_pickle("/home/tnikaein/Documents/PhD/PythonProjects/Netherland/Outputs/DSSAT_code_output/Rain.pkl")
# Rain.index = Rain.index.date
# Rain
Rain=pd.read_pickle(f"/home/tnikaein/Documents/PhD/PythonProjects/Netherland_barabant/{(year)}/Results/Rain.pkl")
Rain.index = Rain.index.date
#___________________________________________________________________________________________________________________________________________

# radar features
Amp_VV_New=pd.read_pickle(f"/home/tnikaein/Documents/PhD/PythonProjects/Netherland_barabant/Sandbox/{(YY)}/Out_For_DSSAT/Amp18_VV_New.pkl")


Amp_VH_New=pd.read_pickle(f"/home/tnikaein/Documents/PhD/PythonProjects/Netherland_barabant/Sandbox/{(YY)}/Out_For_DSSAT/Amp18_VH_New.pkl")


Amp_CR_New=pd.read_pickle(f"/home/tnikaein/Documents/PhD/PythonProjects/Netherland_barabant/Sandbox/{(YY)}/Out_For_DSSAT/Amp18_CR_New.pkl")

##___________________________________________________________________________________________________________________________________________
# finding the time that both parameters are available (LAI is daily but AMP is not)
harvestdate=datetime.date(YY, 10, 1)
IDX2 = Brabant_LAI.index.intersection(Amp_VV_New.index)
IDX=IDX2[IDX2<harvestdate]
field_ID =  Brabant_LAI.columns.intersection(Amp_VV_New.columns)


#IDX = Brabant_SWTD.index.intersection(Amp_VV_New.index)  # only if use swtd

Amp_VV_New2 = Amp_VV_New.loc[IDX,field_ID].T
Amp_VH_New2 = Amp_VH_New.loc[IDX,field_ID].T
Amp_CR_New2 = Amp_CR_New.loc[IDX,field_ID].T

Brabant_LAI2 = Brabant_LAI.loc[IDX,field_ID].T
Brabant_SWTD2 = Brabant_SWTD.loc[IDX,field_ID].T
Brabant_SWTD62 = Brabant_SWTD6.loc[IDX,field_ID].T
Brabant_CHTD2 = Brabant_CHTD.loc[IDX,field_ID].T
Brabant_CWAD2 = Brabant_CWAD.loc[IDX,field_ID].T
Rain2 = Rain.loc[IDX,field_ID].T
#___________________________________________________________________________________________________________________________________________
# Hold out part of the data
Train_F_NO = 1000
Amp_VV_hold = Amp_VV_New2.iloc[Train_F_NO:,:]
Amp_VV_rest = Amp_VV_New2.iloc[:Train_F_NO,:]
Amp_VH_hold = Amp_VH_New2.iloc[Train_F_NO:,:]
Amp_VH_rest = Amp_VH_New2.iloc[:Train_F_NO,:]
Amp_CR_hold = Amp_CR_New2.iloc[Train_F_NO:,:]
Amp_CR_rest = Amp_CR_New2.iloc[:Train_F_NO,:]

Brabant_LAI2_hold = Brabant_LAI2.iloc[Train_F_NO:,:]
Brabant_LAI2_rest = Brabant_LAI2.iloc[:Train_F_NO,:]

Brabant_SWTD2_hold = Brabant_SWTD2.iloc[Train_F_NO:,:]
Brabant_SWTD2_rest = Brabant_SWTD2.iloc[:Train_F_NO,:]

Brabant_SWTD62_hold = Brabant_SWTD62.iloc[Train_F_NO:,:]
Brabant_SWTD62_rest = Brabant_SWTD62.iloc[:Train_F_NO,:]

Brabant_CHTD2_hold = Brabant_CHTD2.iloc[Train_F_NO:,:]
Brabant_CHTD2_rest = Brabant_CHTD2.iloc[:Train_F_NO,:]

Brabant_CWAD2_hold = Brabant_CWAD2.iloc[Train_F_NO:,:]
Brabant_CWAD2_rest = Brabant_CWAD2.iloc[:Train_F_NO,:]

Rain2_hold = Rain2.iloc[Train_F_NO:,:]
Rain2_rest = Rain2.iloc[:Train_F_NO,:]

#___________________________________________________________________________________________________________________________________________
R3 = Rain2_rest.values.flatten()
R3_H = Rain2_hold.values.flatten()

LAI3 = Brabant_LAI2_rest.values.flatten()
LAI3_H = Brabant_LAI2_hold.values.flatten()

SWTD_L1 = Brabant_SWTD2_rest.values.flatten()
SWTD_L1_H = Brabant_SWTD2_hold.values.flatten()

SWTD_L6 = Brabant_SWTD62_rest.values.flatten()
SWTD_L6_H = Brabant_SWTD62_hold.values.flatten()

CHTD3 = Brabant_CHTD2_rest.values.flatten()
CHTD3_H = Brabant_CHTD2_hold.values.flatten()

CWAD3 = Brabant_CWAD2_rest.values.flatten()
CWAD3_H = Brabant_CWAD2_hold.values.flatten()

Amp_VV_New3 = Amp_VV_rest.values.flatten()
Amp_VV_New3_h = Amp_VV_hold.values.flatten()

Amp_VH_New3 = Amp_VH_rest.values.flatten()
Amp_VH_New3_h = Amp_VH_hold.values.flatten()

Amp_CR_New3 = Amp_CR_rest.values.flatten()
Amp_CR_New3_h = Amp_CR_hold.values.flatten()


#___________________________________________________________________________________________________________________________________________
# regression
X = np.vstack((LAI3,SWTD_L1,SWTD_L6,CWAD3)).T
X_H =np.vstack((LAI3_H,SWTD_L1_H,SWTD_L6_H,CWAD3_H)).T
#X = SWTD3
#X_H = SWTD3_H

Y = 10*np.log10(Amp_CR_New3)
Y_H = 10*np.log10(Amp_CR_New3_h)
#_____________________________________________________________________________________________________________________
# # save data to use by fsrmrmr in matlab for feature ranking
tbl = np.column_stack((X,Y))
tbll=pd.DataFrame(tbl)
tbll.columns = ["LAI","SM_s","SM_rz","TW","class"]
tbll.to_csv(f"/home/tnikaein/Documents/PhD/PythonProjects/Netherland_barabant/{(year)}/FSRMRMR/tbl_AMPCR_Visualization.csv",header=True,index=False)
#_____________________________________________________________________________________________________________________
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.2)

#SVR --------------------------------------------
# Tuning
regSVR = make_pipeline(MinMaxScaler(),SVR())
kernel = ["poly","rbf","sigmoid"]
C = [100,10,1,0.1]
gamma = ["scale"]
# define grid search
grid = dict(svr__kernel=kernel,svr__C=C,svr__gamma=gamma)
cv = RepeatedKFold(n_splits=4,n_repeats=2,random_state=1)
grid_search = GridSearchCV(estimator=regSVR, param_grid=grid,n_jobs=-1,cv=cv,scoring=["r2","neg_mean_squared_error"],refit="r2")
grid_result = grid_search.fit(X,Y)
#########################################################################################################################
pkl_filename = f"/home/tnikaein/Documents/PhD/PythonProjects/Netherland_barabant/{(year)}/Results/Model_pkl/{(YY)}_CR.pkl"
###save model###
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(grid_result, file)
#########################################################################################################################
###load model###
with open(pkl_filename, 'rb') as file:
    grid_result = pickle.load(file)
#########################################################################################################################
Y_eval = grid_result.predict(X_H)
Y_eval_file_name = f"/home/tnikaein/Documents/PhD/PythonProjects/Netherland_barabant/{(year)}/Results/Model_pkl/Y_EVAL{(YY)}_CR.pkl"
with open(Y_eval_file_name, 'wb') as file_y:
     pickle.dump(Y_eval,file_y)

MSE_SVR = mean_squared_error(Y_H,Y_eval)
MAE_SVR = mean_absolute_error(Y_H,Y_eval)
R_2 = r2_score(Y_H,Y_eval)
pearsonr_SVR_AM,_= pearsonr(Y_H,Y_eval)
spearmanr_SVR_AM,_= spearmanr(Y_H,Y_eval)

print("MAE:%f   MSE:%f  R2:%f   PEARSONR:%f  SPEARMANR:%f" %(MAE_SVR,MSE_SVR,R_2,pearsonr_SVR_AM,spearmanr_SVR_AM))


#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#___________________________________________________________________________________________________________________________________________
# calculating correlation indices for each field between estimated backscatter and observable
Y_eval_linear=10**(Y_eval/10)
Y_eval_reshape=Y_eval_linear.reshape(283,24)
Y_eval_reshape_df = pd.DataFrame(Y_eval_reshape)

def correlation_ObsEst_field(data_hold):
    data_hold=data_hold.reset_index(drop=True)
    data_hold.columns = Y_eval_reshape_df.columns
    correl_spearman=data_hold.corrwith(Y_eval_reshape_df,axis=1,method="spearman")
    correl_pearson=data_hold.corrwith(Y_eval_reshape_df,axis=1,method="pearson")
    return correl_spearman,correl_pearson

#  #####----FOR VH-----#######
# [corr_spearman,corr_pearson] = correlation_ObsEst_field(Amp_VH_hold)
# corr_spearman.to_pickle( "/home/tnikaein/Documents/PhD/PythonProjects/Netherland/Outputs/Analysis_output/correl_spearman_VH.pkl")
# corr_pearson.to_pickle("/home/tnikaein/Documents/PhD/PythonProjects/Netherland/Outputs/Analysis_output/correl_pearson_VH.pkl")

# #####----FOR VV-----#######
# [corr_spearman,corr_pearson] = correlation_ObsEst_field(Amp_VV_hold)
# corr_spearman.to_pickle( "/home/tnikaein/Documents/PhD/PythonProjects/Netherland/Outputs/Analysis_output/correl_spearman_VV.pkl")
# corr_pearson.to_pickle("/home/tnikaein/Documents/PhD/PythonProjects/Netherland/Outputs/Analysis_output/correl_pearson_VV.pkl")

######----FOR CR-----#######
[corr_spearman,corr_pearson] = correlation_ObsEst_field(Amp_CR_hold)
corr_spearman.to_pickle( f"/home/tnikaein/Documents/PhD/PythonProjects/Netherland_barabant/{(year)}/Results/correl_spearman_VV.pkl")
corr_pearson.to_pickle(f"/home/tnikaein/Documents/PhD/PythonProjects/Netherland_barabant/{(year)}/Results/correl_pearson_VV.pkl")

#____________________________________
# Apply generic model on the single field
def Apply_generic_ON_Single_Field(AMP_hold,NOfield,title,):
    fig, ax = plt.subplots(figsize=(16, 4))
    plt.plot(IDX,10*np.log10(AMP_hold.iloc[NOfield]),"b",label="Observed")
    plt.plot(IDX,10*np.log10(Y_eval_reshape_df.iloc[NOfield]),"r",label="Estimated")
    plt.legend()
    plt.ylabel("[dB] ", fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Time", fontsize=16)
    plt.xticks(fontsize=12)
    plt.title("Amplitude "+title+" generic model vs one test field", fontsize=16)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    plt.grid()
    plt.tight_layout()


Apply_generic_ON_Single_Field(Amp_VH_hold,100,"VH")
#Apply_generic_ON_Single_Field(Amp_VV_hold,100,"VV")
#Apply_generic_ON_Single_Field(Amp_CR_hold,100,"CR")

#___________________________________________________________________________________________________________________________________________
#OVER TEST DATA
# taking average over fields in linear domain for estimated and observed backscatter to look at how close they are
Y_eval_linear=10**(Y_eval/10)
#Y_eval_reshape=Y_eval_linear.reshape(283,24)#2017
#Y_eval_reshape=Y_eval_linear.reshape(268,25)#2018
Y_eval_reshape=Y_eval_linear.reshape(257,25)#2019

Y_eval_estimated = np.mean(Y_eval_reshape,axis=0)
Avg_Amp_VV = np.mean(Amp_VV_hold,axis=0)
Avg_Amp_VH = np.mean(Amp_VH_hold,axis=0)
Avg_Amp_CR = np.mean(Amp_CR_hold,axis=0)

## percentiles 20
Y_eval_estimated20 = np.percentile(Y_eval_reshape,20,axis=0)
Amp_perc_CR20 = np.percentile(Amp_CR_hold,20, axis=0)
Amp_perc_VV20 = np.percentile(Amp_VV_hold,20, axis=0)
Amp_perc_VH20 = np.percentile(Amp_VH_hold,20, axis=0)

## percentiles 80
Amp_perc_CR80 = np.percentile(Amp_CR_hold,80, axis=0)
Amp_perc_VV80 = np.percentile(Amp_VV_hold,80, axis=0)
Amp_perc_VH80 = np.percentile(Amp_VH_hold,80, axis=0)
Y_eval_estimated80 = np.percentile(Y_eval_reshape,80,axis=0)

def plot_AMP_TESTData(AVG_AMPLITUDE,AVG_AMPLITUDE20,AVG_AMPLITUDE80,title,ymin,ymax ):
    fig, ax = plt.subplots(figsize=(7, 3.7))
    plt.plot(IDX,10*np.log10(AVG_AMPLITUDE),"b",label="Observed")
    plt.fill_between(IDX, 10*np.log10(AVG_AMPLITUDE20), 10*np.log10(AVG_AMPLITUDE80),alpha=0.2,  edgecolor="b", facecolor="b", linewidth=0)
    plt.plot(IDX,10*np.log10(Y_eval_estimated),"r",label="Estimated")
    plt.fill_between(IDX, 10*np.log10(Y_eval_estimated20), 10*np.log10(Y_eval_estimated80),alpha=0.2,  edgecolor="r", facecolor="r", linewidth=0)
    #plt.legend()
    plt.ylabel(title+"[dB]", fontsize=12)
    plt.yticks(fontsize=12)
    #plt.xlabel("Time", fontsize=16)
    plt.xticks(fontsize=12)
    #plt.title(title, fontsize=16)
    #ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=WE, interval=3)) #2019 WE #2018 TU #2017 mo
    #ax.xaxis.set_major_locator(mdates.MonthLocator())
    #ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonthday=15))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    plt.ylim(ymin,ymax)
    #ax.set_xlim([datetime.date(2017, 4, 30), datetime.date(2017, 10, 5)])
    #ax.set_xlim([datetime.date(2018, 4, 30), datetime.date(2018, 10, 5)])
    ax.set_xlim([datetime.date(2019, 4, 30), datetime.date(2019, 10, 5)])
    plt.grid()
    #plt.tight_layout()
    plt.subplots_adjust(left=0.07, bottom=0.07, right=0.98, top=0.95, wspace=None, hspace=None)


plot_AMP_TESTData(Avg_Amp_VH,Amp_perc_VH20,Amp_perc_VH80,"VH",-26,-14)
plt.savefig(F"/home/tnikaein/Documents/PhD/PythonProjects/Netherland_barabant/{(year)}/Results/testdata/VH.png",dpi=500)

plot_AMP_TESTData(Avg_Amp_VV,Amp_perc_VV20,Amp_perc_VV80,"VV",-15.5,-7.9)
plt.savefig(F"/home/tnikaein/Documents/PhD/PythonProjects/Netherland_barabant/{(year)}/Results/testdata/VV.png",dpi=500)

plot_AMP_TESTData(Avg_Amp_CR,Amp_perc_CR20,Amp_perc_CR80,"CR",-11,-3.9)
plt.savefig(F"/home/tnikaein/Documents/PhD/PythonProjects/Netherland_barabant/{(year)}/Results/testdata/CR.png",dpi=500)

#####################################################################

ALL_IDX = (Brabant_LAI.T).columns
plot_IDX=ALL_IDX[ALL_IDX<harvestdate]
def plot_AMP_TestData_with_inputs(AVG_AMPLITUDE,title):

    fig,ax = plt.subplots(6,sharex='col')
    ax[0].plot(IDX,10*np.log10(AVG_AMPLITUDE),"r")
    ax[0].set_ylabel("Amp S1 "+title+"[dB]",rotation=0,labelpad=60,fontsize=12)
    ax[0].tick_params(axis='y',labelsize=12)

    ax[1].plot(IDX,10*np.log10(Y_eval_estimated),"r")
    ax[1].set_ylabel("Amp Est " + title + "[dB]", rotation=0, labelpad=60, fontsize=12)
    ax[1].tick_params(axis='y', labelsize=12)

    ax[2].plot(plot_IDX[1:],np.mean(((Brabant_LAI.T).iloc[Train_F_NO:,1:148]),axis=0),"g")
    ax[2].set_ylabel("LAI", rotation=0, labelpad=40, fontsize=12)
    ax[2].tick_params(axis='y', labelsize=12)

    ax[3].plot(plot_IDX[1:],np.mean(((Brabant_SWTD.T).iloc[Train_F_NO:,127:274]),axis=0),"b")
    ax[3].set_ylabel("SM[%]", rotation=0, labelpad=40, fontsize=12)
    ax[3].tick_params(axis='y', labelsize=12)

    ax[4].plot(plot_IDX[1:],np.mean(((Brabant_SWTD6.T).iloc[Train_F_NO:,127:274]),axis=0),"b")
    ax[4].set_ylabel("SM[%]", rotation=0, labelpad=40, fontsize=12)
    ax[4].tick_params(axis='y', labelsize=12)

    ax[5].plot(plot_IDX[1:],np.mean(((Brabant_CWAD.T).iloc[Train_F_NO:,1:148]),axis=0),"k")
    ax[5].set_ylabel("Dry biomass[kg/m2]", rotation=0, labelpad=60, fontsize=12)
    ax[5].tick_params(axis='y', labelsize=12)

    # ax[6].plot(plot_IDX[1:],np.mean(((Brabant_CHTD.T).iloc[Train_F_NO:,1:148]),axis=0),'y')
    # ax[6].set_ylabel("Height[m]", rotation=0, labelpad=40, fontsize=14)
    ax[5].set_xlabel("Time", fontsize=14)
    ax[5].tick_params(axis='y', labelsize=12)
    ax[5].tick_params(axis='x', labelsize=12)

    fig.align_ylabels()
    plt.suptitle(title)


plot_AMP_TestData_with_inputs(Avg_Amp_VH,"VH")
plot_AMP_TestData_with_inputs(Avg_Amp_VV,"VV")
plot_AMP_TestData_with_inputs(Avg_Amp_CR,"CR")

#___________________________________________________________________________________________________________________________________________
##these plots are independant from regression results

def Model_inputs_plot(data,color,ylabel,mode):
    if mode=="P":
     data = (data.T).iloc[Train_F_NO:,0:148]
    if mode=="W":
     data= (data.T).iloc[Train_F_NO:,126:274]
    if mode== "R":
     data = (data.T).iloc[Train_F_NO:,125:273]


    fig,ax= plt.subplots(figsize=(16,4))
    plt.plot(data.columns,np.mean(data,axis=0),color=color)
    a= np.mean(data,axis=0)-np.std(data,axis=0)
    b= np.mean(data,axis=0)+np.std(data,axis=0)

    plt.fill_between(data.columns,a,b,color=color,alpha=0.2)
    plt.xlabel("Time",fontsize=16)
    plt.ylabel(ylabel,fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=16)
    #plt.legend(fontsize=16)
    #loc = plticker.MultipleLocator(base=5)
    #ax.yaxis.set_major_locator(loc)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    plt.grid()
    plt.tight_layout()


Model_inputs_plot(Brabant_LAI,"g","LAI","P")
Model_inputs_plot(Brabant_CHTD,"k","Height[m]","P")
Model_inputs_plot(Brabant_CWAD,"r","Dry biomass[kg/m$^{2}$]","P")
Model_inputs_plot(Brabant_SWTD,"b","SM[%]","W")
plt.ylim(0,0.5)
Model_inputs_plot(Brabant_SWTD6,"b","SM[%]","W")
Model_inputs_plot(Rain,"c","Rain[mm]","R")

#___________________________________________________________________________________________________________________________________________
#scatter plot for separated test data with considering rain

fig, axes = plt.subplots(nrows=3,ncols=4,sharex='col',sharey='row')
sns.scatterplot(x=LAI3,y=10*np.log10(Amp_VV_New3),ax=axes[0,0],hue=R3,size=R3,palette=sns.color_palette('viridis', as_cmap = True),   legend=False)
sns.scatterplot(x=SWTD_L1,y=10*np.log10(Amp_VV_New3),ax=axes[0,1],hue=R3,size=R3,palette=sns.color_palette('viridis', as_cmap = True),   legend=False)
sns.scatterplot(x=CHTD3,y=10*np.log10(Amp_VV_New3),ax=axes[0,2],hue=R3,size=R3,palette=sns.color_palette('viridis', as_cmap = True),   legend=False)
sns.scatterplot(x=CWAD3,y=10*np.log10(Amp_VV_New3),ax=axes[0,3],hue=R3,size=R3,palette=sns.color_palette('viridis', as_cmap = True),   legend=False)
axes[0,0].set_ylabel("Amp VV[dB]",fontsize=14)


sns.scatterplot(x=LAI3,y=10*np.log10(Amp_VH_New3),ax=axes[1,0],hue=R3,size=R3,palette=sns.color_palette('viridis', as_cmap = True),   legend=False)
sns.scatterplot(x=SWTD_L1,y=10*np.log10(Amp_VH_New3),ax=axes[1,1],hue=R3,size=R3,palette=sns.color_palette('viridis', as_cmap = True),   legend=False)
sns.scatterplot(x=CHTD3,y=10*np.log10(Amp_VH_New3),ax=axes[1,2],hue=R3,size=R3,palette=sns.color_palette('viridis', as_cmap = True),   legend=False)
sns.scatterplot(x=CWAD3,y=10*np.log10(Amp_VH_New3),ax=axes[1,3],hue=R3,size=R3,palette=sns.color_palette('viridis', as_cmap = True),   legend=False)
axes[1,0].set_ylabel("Amp VH[dB]",fontsize=14)

sns.scatterplot(x=LAI3,y=10*np.log10(Amp_CR_New3),ax=axes[2,0],hue=R3,size=R3,palette=sns.color_palette('viridis', as_cmap = True),   legend=False)
sns.scatterplot(x=SWTD_L1,y=10*np.log10(Amp_CR_New3),ax=axes[2,1],hue=R3,size=R3,palette=sns.color_palette('viridis', as_cmap = True),   legend=False)
sns.scatterplot(x=CHTD3,y=10*np.log10(Amp_CR_New3),ax=axes[2,2],hue=R3,size=R3,palette=sns.color_palette('viridis', as_cmap = True),   legend=False)
sns.scatterplot(x=CWAD3,y=10*np.log10(Amp_CR_New3),ax=axes[2,3],hue=R3,size=R3,palette=sns.color_palette('viridis', as_cmap = True),   legend="brief")
axes[2,0].set_ylabel("Amp CR[dB]",fontsize=14)


axes[2,0].set_xlabel("LAI",fontsize=14)
axes[2,1].set_xlabel("SM[%]",fontsize=14)
axes[2,2].set_xlabel("Height[m]",fontsize=14)
axes[2,3].set_xlabel("Top Weight[kg/m$^{2}$]",fontsize=14)
plt.suptitle("Observed VS inputs for train maize fields")

lines = []
labels = []
for ax in fig.axes:
    axLine, axLabel = ax.get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)
fig.legend(lines, labels,
           loc='upper right',title="Rain",fontsize=14,title_fontsize=16)
axes[2,3].legend().remove()
plt.show()

#___________________________________________________________________________________________________________________________________________

# in a case of ploting with dual y axis

# fig, ax1 = plt.subplots()
# color = 'tab:red'
# ax1.plot(IDX,10*np.log10(Avg_Amp_VH),"r",label="Observed")
# ax1.plot(IDX,10*np.log10(Y_eval_estimated),"r:",label="Estimated")
# ax1.set_xlabel('time (s)')
# ax1.set_ylabel('Amplitude [dB]', color=color)
#
#
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# color = 'tab:blue'
# ax2.plot(IDX,np.mean(Brabant_SWTD2_hold,axis=0),"b--",label="SM")
# ax2.set_ylabel("Soil moisture[%]",color=color)
# fig.legend(loc="upper right",bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)