#conditions (in this case, predict outcome of non-remission based on cdai)
outcome_index = 2
feature_in_model = 'ALL_features'
repeat_times = 100

#all features,outcomes
category_clinical_varibles = ['sex', 'smoking', 'anti_ccp', 'RF_pos_tot']
continous_clinical_varibles = ['age', 'bmi','durmonths', 'DASc_0', 'sdai_0',
                              'cdai_0', 'pga_0', 'phga_0', 'sjc44_0', 'trs_0',
                              'esr_0', 'crp_0', 'pfts','fatigue_0','jointpain_0',
                               'raid_tot_0', 'dose_mtx_0']
outcome =  ['acreular_rem_6mnts', 'sdairem_6mnts', 'cdairem_6mnts', 'respgood_4mnts', 'outcome12and24binary']#
criteria_c = outcome[outcome_index]

# Commented out IPython magic to ensure Python compatibility.
#import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, auc, classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.svm import SVC
!pip install tqdm
from tqdm import tqdm

#read original dataset
df = pd.read_csv('')
df_ml = df[category_clinical_varibles + continous_clinical_varibles + [criteria_c] + ['sampleID']]
df_ml.shape

## make the non-remission as 1 and remission as 0
df_ml[criteria_c] = [1-i for i in df_ml[criteria_c]]

###function for make dummies
def dummy_categorical(df_input, list_cate_feature):
        df_categorical_dummy = pd.get_dummies(df_input, columns = list_cate_feature,
                                         drop_first = True)
        return df_categorical_dummy

##make dummies for categorical variables
df_dummied = dummy_categorical(df_ml, category_clinical_varibles)
df_dummied.shape
df_dummied_resetindex = df_dummied.set_index('sampleID')

X_ml = df_dummied_resetindex.drop([criteria_c], axis = 1) #(222, 22) 21 features plus sampleID
y_ml = df_dummied[criteria_c] #(222, 1)

y_ml.value_counts()

############prepare some functions###################
def calculate_confusion_matrix(groundtruth, predicted_class):
  df_test = pd.DataFrame({'truth': groundtruth, 'pred' : predicted_class})
  tp = df_test[(df_test['truth'] == 1) & (df_test['pred'] == 1)].shape[0]
  tn = df_test[(df_test['truth'] == 0) & (df_test['pred'] == 0)].shape[0]
  fp = df_test[(df_test['truth'] == 0) & (df_test['pred'] == 1)].shape[0]
  fn = df_test[(df_test['truth'] == 1) & (df_test['pred'] == 0)].shape[0]
  precision_p = tp/(tp + fp)
  precision_n = tn /(tn +fn)
  return (tp,tn,fp,fn), precision_p, precision_n

def standard(df_input, contin_col):
  scaler = StandardScaler()
  df_imputed_array = scaler.fit_transform(df_input[contin_col])
  df_imputed = pd.DataFrame(df_imputed_array, columns = df_input[contin_col].columns)
  df_input[contin_col] = df_imputed[contin_col]
  df_output = df_input
  return df_output

def impute_standard(df_input):
  imputer= KNNImputer(n_neighbors=5, weights="uniform")
  df_imputed_array = imputer.fit_transform(df_input)
  df_output = pd.DataFrame(df_imputed_array, columns = df_input.columns)
  df_output_stand = standard(df_output, continous_clinical_varibles)
  return df_output

###start tuning and training and testing

#define model and cv
cv_inner = StratifiedKFold(n_splits=5, shuffle = True)
model = LogisticRegression()
model_rfc = RandomForestClassifier()
model_svm = SVC(probability = True)

#hyperparameter tuned during training
param_grid = [{'C': [0.01, 0.03, 0.06, 0.1,0.12,0.14, 0.16, 0.18, 0.2, 0.3,0.4,0.5,0.6, 0.7, 0.8,0.9, 1, 5, 10, 100],
              'penalty': ['elasticnet'],
              'solver': ['saga'],
              'l1_ratio':[0.001,0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.999],
               'max_iter': [5000]}]

param_grid_rfc = {'max_depth': [10, 20, 30],
              'n_estimators': [100, 125, 150],
              'min_samples_split': [3, 4],
              'max_features' : ['sqrt']}

param_grid_svm = {'C':[0.1, 0.5, 1, 10], 'gamma':['scale', 'auto'],
                'kernel':['linear', 'poly','sigmoid']}

list_f1 = []
list_roc_result = []
list_roc_result_rfc = []
list_roc_result_svm = []


dic_result_allsplits_evaluation = {}
dic_result_allsplits_modelinfo = {}
dic_roc_auc_allsplits = {}
dic_rocauccurve_allsplits = {}


round = 0
for i in tqdm(range(100)):
  round +=1
  X_train_outer_pre,X_test_pre,y_train_outer,y_test = train_test_split(X_ml, y_ml, test_size=0.3)

  #imputate the missing data for train and test seperately
  #X_test and y_test will be hold-out for final testing
  #X_train_outer and y_train_outer used for model training by inner cv & grid-search
  X_train_outer = impute_standard(X_train_outer_pre)
  X_test = impute_standard(X_test_pre)

  #training model in the train_outer by elastic net
  grid_search_inner = GridSearchCV(model, param_grid, cv = cv_inner,
                           scoring='roc_auc',n_jobs=-1)
  grid_result_inner= grid_search_inner.fit(X_train_outer, y_train_outer)
  best_params_inner = grid_search_inner.best_params_

  #evaluate the trained model in test set
  #calculate the probability of getting approved
  #measure the auc-roc
  y_test_prediction = grid_result_inner.predict_proba(X_test)[:,1]
  y_test = np.ravel(y_test)
  roc_score = roc_auc_score(y_test,y_test_prediction)
  list_roc_result.append(roc_score)

  #repeat the above training and evaluation process for rfc and svm respectively
  grid_search_inner_rfc = GridSearchCV(model_rfc, param_grid_rfc, cv = cv_inner,
                           scoring='roc_auc',n_jobs=-1)
  grid_result_inner_rfc = grid_search_inner_rfc.fit(X_train_outer, y_train_outer)
  y_test_prediction_rfc = grid_result_inner_rfc.predict_proba(X_test)[:,1]
  roc_score_rfc = roc_auc_score(y_test,y_test_prediction_rfc)
  list_roc_result_rfc.append(roc_score_rfc)


  grid_search_inner_svm = GridSearchCV(model_svm, param_grid_svm, cv = cv_inner,
                           scoring='roc_auc',n_jobs=-1)
  grid_result_inner_svm = grid_search_inner_svm.fit(X_train_outer, y_train_outer)
  y_test_prediction_svm = grid_result_inner_svm.predict_proba(X_test)[:,1]
  roc_score_svm = roc_auc_score(y_test,y_test_prediction_svm)
  list_roc_result_svm.append(roc_score_svm)


  #######all related analysis for elastic net algorithm are as below######

  ## 1. calculate feature importance in the whole dataset (X, y)
  X_ml_trans = impute_standard(X_ml)
  modelfinal = grid_search_inner.best_estimator_.fit(X_ml_trans, y_ml)
  coef = modelfinal.coef_
  coef_flat = [item for sublist in coef for item in sublist]
  coeff_df = pd.DataFrame(coef_flat,X_ml_trans.columns,columns=['Coefficient'])

  ## 2. calculate fpr, tpr, thresholds
  fpr, tpr, thresholds = roc_curve(y_test, y_test_prediction)
  thresholds[0] = thresholds[0] - 1
  df_fpr_tpr = pd.DataFrame({'FPR':fpr, 'TPR':tpr, 'Threshold':thresholds})
  df_fpr_tpr['sensitivity'] = df_fpr_tpr['TPR']
  df_fpr_tpr['specificity'] = 1-df_fpr_tpr['FPR']

  ## 3. according on the best f1_macro score to set the thresholds
  f1_scores_list = [f1_score(y_test, [int(p >= t) for p in y_test_prediction], average = 'macro') for t in thresholds] #calculate f1 scores based on different thresholds
  threshold_max_f1 = thresholds[np.argmax(f1_scores_list)]
  y_test_prediction_reclassified_f1 = [1 if i >= threshold_max_f1 else 0 for i in y_test_prediction]
  sen_best_f1 = df_fpr_tpr['sensitivity'].tolist()[thresholds.tolist().index(threshold_max_f1)]
  spe_best_f1 = df_fpr_tpr['specificity'].tolist()[thresholds.tolist().index(threshold_max_f1)]
  f1_max = max(f1_scores_list)
  tuple_confusion_matrix_f1, precision_p_f1, precision_n_f1  = calculate_confusion_matrix(y_test, y_test_prediction_reclassified_f1) #.values.flatten()

  ## result of threshold related metrics
  dic_f1 = {'threshold_max_f1': threshold_max_f1,
            'sen_best_f1': sen_best_f1,
            'spe_best_f1': spe_best_f1,
            'f1_max': f1_max,
            'tuple_confusion_matrix_f1': tuple_confusion_matrix_f1,
            'precision_p_f1':precision_p_f1,
            'precision_n_f1': precision_n_f1
            }
  ##result of hyperparameters
  dic_modelinfo = {'hyper_parameter': best_params_inner, 'feature_importance':coeff_df
                   }
  dic_result_allsplits_modelinfo[round] = dic_modelinfo

  ##result of roc curve
  dic_rocauccurve = {'fpr': fpr, 'tpr':tpr, 'thresholds': thresholds}
  dic_rocauccurve_allsplits[round] = dic_rocauccurve

##############data analysis for roc score in elastic net########################

df_roc_auc = pd.DataFrame(list_roc_result, columns = ['roc_auc'])
df_roc_auc.index = [f'split{i+1}' for i in range(len(df_roc_auc))]
df_roc_auc.loc['mean'] =df_roc_auc['roc_auc'].mean()

print('Mean:{}\nSD:{}'.format(np.mean(list_roc_result), np.std(list_roc_result)))

############################draw roc curve#####################################

sns.set_theme(style="white")
colors = {1: '#B0AC9D',
          2: '#FFA07A',
         3: '#FF5733',
        4: '#8EAC4E',
        5: '#D7899B'}
plt.rcParams['font.family'] = 'DejaVu Sans' #DejaVu Sans cmsy10 Liberation Sans
fig, ax = plt.subplots(figsize=(8,6))

tprs = []
mean_fpr = np.linspace(0, 1, 100)

tpr_list =[]
roc_list = []
fpr_list = []


###draw plot for each split
for i in range(repeat_times-1):
  round = i + 1
  fpr, tpr, thresholds = dic_rocauccurve_allsplits[round]['fpr'], dic_rocauccurve_allsplits[round]['tpr'], dic_rocauccurve_allsplits[round]['thresholds']
  roc = auc(fpr, tpr)
  plt.plot(fpr, tpr, lw=2,
           color = 'lightblue', alpha = 0.3)
  ##append
  interp_tpr = np.interp(mean_fpr, fpr, tpr)
  '''
  This means that I obtain a function based on fpr and tpr,
  then obtain y based on this function with given mean_fpr
  '''
  interp_tpr[0] = 0.0
  tprs.append(interp_tpr)

  tpr_list.append(tpr)
  fpr_list.append(fpr)
  roc_list.append(roc)

################################################################################
mean_tpr = np.mean(tprs, axis = 0)
mean_tpr[-1] = 1.0
#mean_auc = auc(mean_fpr, mean_tpr)
mean_auc = np.mean(df_dic_roc_auc_allsplits['test_set'].tolist())
std_auc = np.std(roc_list)
##plot the mean curve
plt.plot(mean_fpr, mean_tpr, lw=3,
           label=r"Mean AUC-ROC (%0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
           color = 'darkgreen', alpha = 1)


fpr, tpr, thresholds = dic_rocauccurve_allsplits[100]['fpr'], dic_rocauccurve_allsplits[100]['tpr'], dic_rocauccurve_allsplits[100]['thresholds']
plt.plot(fpr, tpr, lw=2,
           label='all splits',
           color = 'lightblue', alpha = 0.3)

##plot the std
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

##plot the chance level
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label = 'chance level')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 14)
plt.ylabel('True Positive Rate', fontsize = 14)
plt.title('ROC Curve in test set\n(CDAI non-Remission with all features)', fontsize = 18, fontweight='bold')
plt.legend(loc="lower right")
plt.show()

#####################calculate mean from 100 splits##############################

dfs = []
for i in range(100):
  dic_result_allsplits_modelinfo[i+1]['feature_importance'].rename({'Coefficient': 'Coefficient_split{}.format(i+1)'}, inplace = True)
  dfs.append(dic_result_allsplits_modelinfo[i+1]['feature_importance'])

# get the index from one of the dataframes
index = dfs[0].index

# reindex all the dataframes to have the same index
dfs_reordered = [df_t.reindex(index) for df_t in dfs]

# concatenate the dataframes along the axis of the columns
df_concat = pd.concat(dfs_reordered, axis=1)

# calculate the mean value for each index

mean_df = df_concat.mean(axis=1)
df_concat['mean'] = mean_df
df_concat['mean_abs'] = abs(df_concat['mean'])

df_concat_sort = df_concat.sort_values(by = ['mean'], ascending = False)
df_concat_sort['features'] = df_concat_sort.index

########draw draw draw##########

###pre-processing
df_not0 = df_concat_sort[df_concat_sort['mean_abs'] > 0.01 ]
df_not0_sort = df_not0.sort_values(by=['mean'], ascending= False)
n_feature = df_not0_sort.shape[0]

df_not0_sort['features'].tolist()

####use feature name transformer#######
category_clinical_varibles = ['sex', 'smoking', 'anti_ccp', 'RF_pos_tot']
continous_clinical_varibles = ['age', 'bmi','durmonths', 'DASc_0', 'sdai_0',
                              'cdai_0', 'pga_0', 'phga_0', 'sjc44_0', 'trs_0',
                              'esr_0', 'crp_0', 'pfts','fatigue_0','jointpain_0',
                               'raid_tot_0', 'dose_mtx_0']  #

dic_transform_name = {'sex_Male':'Male', 'smoking_Smoker':'Current Smoker', 'anti_ccp': 'ACAP(positive)',
                      'RF_pos_tot': 'RF(positive)', 'age': 'Age', 'bmi': 'BMI',
                      'durmonths': 'During Months', 'DASc_0': 'DAS', 'sdai_0': 'SDAI',
                      'cdai_0': 'CDAI', 'pga_0': 'PGA', 'phga_0': 'PhGA', 'sjc44_0': 'SJC44',
                       'trs_0': 'RAI', 'esr_0': 'ESR', 'crp_0': 'CRP', 'pfts':'PROMIS-PF',
                      'fatigue_0': 'Fatigue','jointpain_0': 'Joint pain',
                               'raid_tot_0': 'RAID Score', 'dose_mtx_0': 'MTX Dose' }

def normalized_name(list_input):
  list_output = [dic_transform_name[i] if i in dic_transform_name.keys() else None for i in list_input]
  return list_output

#custom_palette = sns.diverging_palette(240, 10, n=7, sep = 1, center = 'light')
sns.set_style("white")
list_y_pre = df_not0_sort['features'].tolist()
list_y = normalized_name(list_y_pre)
fig, ax = plt.subplots(figsize=(4,4.5))

plt.axvline(x = 0, ls="dashed", c="grey", lw = 1)    # #A9C9A4 #C6E2FF

#custom_palette = sns.blend_palette(['#E04006', '#BCD2EE'], n_colors=len(list_y), as_cmap=False) #tongue##6ACCBC #BE2625 #A9C9A4 #CC1100
sns.barplot(y=df_not0_sort.index, x=df_not0_sort['mean'], palette='coolwarm_r', ax=ax) #coolwarm_r

y_list = df_not0_sort['mean'].tolist()
y_list_round3 = [np.round(num, 3) for num in y_list]

for i, v in enumerate(y_list_round3):
    if v > 0:
        ax.text(v + 0.01, i + .2, "{:.3f}".format(v), color='black', fontweight='bold')
    else:
        ax.text(v + 0.05*abs(v)*20, i + .2, "{:.3f}".format(v), color='black', fontweight='bold')

plt.yticks(np.arange(0,n_feature,1), list_y, fontsize =12)
plt.title('Feature Importance\n(CDAI non-remission)', fontsize = 14, fontweight='bold')
plt.xlabel('Coefficient', fontsize = 12)
ax.set_xlim(-0.2, 0.8)

#####performance (roc_auc) for 3 algorithms######
dic_list3 = {'EN': list_roc_result,
'RFC': list_roc_result_rfc,
'SVM': list_roc_result_svm}

dic_3algorithms_rocmean = {}
dic_3algorithms_rocstd = {}

for i in range(3):
  mean = np.mean(list(dic_list3.values())[i])
  std = np.std(list(dic_list3.values())[i])
  dic_3algorithms_rocmean[list(dic_list3.keys())[i]] = mean
  dic_3algorithms_rocstd[list(dic_list3.keys())[i]] = std

dic_3algorithms_rocmean

dic_3algorithms_rocstd