#conditions (in this case, predict outcome of non-remission based on cdai)
outcome_index = 2
feature_in_model = 'RAID'
repeat_times = 100

category_clinical_varibles = ['sex', 'smoking', 'anti_ccp', 'RF_pos_tot']
continous_clinical_varibles = ['age', 'bmi','durmonths', 'DASc_0', 'sdai_0',
                              'cdai_0', 'pga_0', 'phga_0', 'sjc44_0', 'trs_0',
                              'esr_0', 'crp_0', 'pfrs_0','fatigue_0','jointpain_0',
                               'raid_tot_0', 'dose_mtx_0']  # . pfts
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
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, auc, classification_report,confusion_matrix,precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.svm import SVC
!pip install tqdm
!pip install plotly
from tqdm import tqdm

#read original dataset
df = pd.read_csv('')
df_ml = df[category_clinical_varibles + continous_clinical_varibles + [criteria_c] + ['sampleID']]
df_ml.shape

## make the non-remission as 1 and remission as 0
df_ml[criteria_c] = [1-i for i in df_ml[criteria_c]]

df_ml_raid = df_Tscore[['raid_tot_0'] + [criteria_c] + ['sampleID']]

df_ml_raid_resetindex = df_ml_raid.set_index('sampleID')

df_ml = df_ml_raid_resetindex.dropna(axis = 0, how = 'any')

X_ml = df_ml.drop([criteria_c], axis = 1)
y_ml = df_ml[criteria_c]

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

###start tuning and training and testing

#define model and cv
cv_inner = StratifiedKFold(n_splits=5, shuffle = True)
model = LogisticRegression()

#hyperparameter tuned during training
param_grid = [{'C': [0.01, 0.03, 0.06, 0.1,0.12,0.14, 0.16,0.18, 0.2, 0.3,0.4,0.5,0.6, 0.7, 0.8,0.9, 1, 5, 10, 100],
              'penalty': ['elasticnet'],
              'solver': ['saga'],
              'l1_ratio':[0.001,0.01, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.999],
               'max_iter': [5000]}]

list_roc_result = []
list_f1 = []

round = 0
dic_result_allsplits_evaluation = {}
dic_result_allsplits_modelinfo = {}
dic_roc_auc_allsplits = {}
dic_rocauccurve_allsplits = {}

for i in tqdm(range(100)):
  round += 1
  X_train_outer_pre,X_test_pre,y_train_outer,y_test = train_test_split(X_ml, y_ml, test_size=0.3)

  #only raid is included, so no need to do data preprocessing
  X_train_outer = X_train_outer_pre
  X_test = X_test_pre

  #training model in the train_outer by elastic net
  grid_search_inner = GridSearchCV(model, param_grid, cv = cv_inner,
                           scoring='roc_auc',n_jobs=-1)
  grid_result_inner= grid_search_inner.fit(X_train_outer, y_train_outer)
  best_params_inner = grid_search_inner.best_params_

  #evaluate the trained model in test set
  #calculate the probability of getting approved
  #measure the auc-roc
  y_test_prediction = grid_result_inner.predict_proba(X_test)[:,1]
  print(f'This is the round{round}')
  y_test = np.ravel(y_test)
  print(y_test_prediction)
  roc_score = roc_auc_score(y_test,y_test_prediction)
  list_roc_result.append(roc_score)

  #calculated fpr, tpr, thresholds
  fpr, tpr, thresholds = roc_curve(y_test, y_test_prediction)
  thresholds[0] = thresholds[0] - 1
  df_fpr_tpr = pd.DataFrame({'FPR':fpr, 'TPR':tpr, 'Threshold':thresholds})
  df_fpr_tpr['sensitivity'] = df_fpr_tpr['TPR']
  df_fpr_tpr['specificity'] = 1-df_fpr_tpr['FPR']

  # according on the best f1_macro score to set the thresholds
  f1_scores_list = [f1_score(y_test, [int(p >= t) for p in y_test_prediction], average = 'macro') for t in thresholds] #calculate f1 scores based on different thresholds
  threshold_max_f1 = thresholds[np.argmax(f1_scores_list)]
  y_test_prediction_reclassified_f1 = [1 if i >= threshold_max_f1 else 0 for i in y_test_prediction]
  sen_best_f1 = df_fpr_tpr['sensitivity'].tolist()[thresholds.tolist().index(threshold_max_f1)]
  spe_best_f1 = df_fpr_tpr['specificity'].tolist()[thresholds.tolist().index(threshold_max_f1)]
  f1_max = max(f1_scores_list)
  tuple_confusion_matrix_f1, precision_p_f1, precision_n_f1  = calculate_confusion_matrix(y_test, y_test_prediction_reclassified_f1) #.values.flatten()
  f1_binary = f1_score(y_test, y_test_prediction_reclassified_f1, average='binary')

  #add to dic
  dic_modelinfo = {'hyper_parameter': best_params_inner}
  dic_result_allsplits_modelinfo[round] = dic_modelinfo

  ##rocinfo
  dic_rocauccurve = {'fpr': fpr, 'tpr':tpr, 'thresholds': thresholds}
  dic_rocauccurve_allsplits[round] = dic_rocauccurve

  #metrics
  dic_f1 = {'threshold_max_f1': threshold_max_f1,
            'sen_best_f1': sen_best_f1,
            'spe_best_f1': spe_best_f1,
            'f1_max': f1_max,
            'f1_binary': f1_binary,
            'tuple_confusion_matrix_f1': tuple_confusion_matrix_f1,
            'precision_p_f1':precision_p_f1,
            'precision_n_f1': precision_n_f1}
  dic_result_onesplit_evaluation = {'f1':dic_f1}
  dic_result_allsplits_evaluation[round] = dic_result_onesplit_evaluation #

list_roc_result
df_roc_auc = pd.DataFrame(list_roc_result, columns = ['roc_auc'])
df_roc_auc.index = [f'split{i+1}' for i in range(len(df_roc_auc))]
df_roc_auc.loc['mean'] =df_roc_auc['roc_auc'].mean()

print('Mean:{}\nSD:{}'.format(np.mean(list_roc_result), np.std(list_roc_result)))

############################draw roc curve#####################################

repeat_times = 100

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
           #label='Split %d (AUC-ROC:%0.2f)' % (round,roc),
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
           color = 'Salmon', alpha = 1)


###
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
plt.title('ROC Curve in test set\n(CDAI non-remission with RAID)', fontsize = 18, fontweight='bold')
plt.legend(loc="lower right")
plt.show()

df_metric_mean = pd.DataFrame()
for i in ['f1_max', 'f1_binary', 'sen_best_f1', 'spe_best_f1', 'precision_p_f1', 'threshold_max_f1']:
  f1_list = [dic_result_allsplits_evaluation[round_num]['f1'][i] for round_num in sorted(dic_result_allsplits_evaluation.keys())]
  mean = np.mean(f1_list)
  std = np.std(f1_list)
  df_metric_mean[i] = [mean, std]
df_metric_mean

#############################hyper_parameter####################################

hyperparams_list_ = []
for i in range(repeat_times):
  hyperparams_list_.append(dic_result_allsplits_modelinfo[i+1]['hyper_parameter'])
df_hyperpara = pd.DataFrame(hyperparams_list_)
df_hyperpara
df_hyperpara.index = [f'split{i+1}' for i in range(len(df_hyperpara))]
df_hyperpara
df_hyperpara['params_tuple'] = df_hyperpara.apply(lambda row: (row['C'], row['l1_ratio'], row['max_iter'], row['penalty'], row['solver']), axis=1)
df_hyperpara

from collections import Counter
best_params_tuple = df_hyperpara['params_tuple'].tolist()
result = Counter(best_params_tuple)
df_count_para = pd.DataFrame.from_dict(result, orient='index', columns=['frequency']).sort_values(by = ['frequency'], ascending=False).head(5)
df_count_para

#####final model using hyperparameters
logit_final = LogisticRegression(penalty='elasticnet', max_iter=5000, solver='saga', C = 0.01, l1_ratio=0.001)
logit_final.fit(X_ml,y_ml)

roc_auc_score(y_ml, logit_final.predict_proba(X_ml)[:,1])

########metrics for final model#######

# 1. Compute predicted probabilities
y_prob = logit_final.predict_proba(X_ml)[:,1]

# 2. Compute metrics across thresholds
thresholds = sorted(y_prob)
sensitivities = []
precisions = []
specificities = []
f1_scores = []

for t in thresholds:
    y_pred = [1 if p >= t else 0 for p in y_prob]
    tn, fp, fn, tp = confusion_matrix(y_ml, y_pred).ravel()

    sensitivity = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

    sensitivities.append(sensitivity)
    precisions.append(precision)
    specificities.append(specificity)
    f1_scores.append(f1)

fig = go.Figure()

fig.add_trace(go.Scatter(x=thresholds, y=sensitivities, mode='lines', name='Sensitivity', line = dict(color = 'rgb(31, 119, 180)')))
fig.add_trace(go.Scatter(x=thresholds, y=precisions, mode='lines', name='Precision', line=dict(color = "#FA8072")))
fig.add_trace(go.Scatter(x=thresholds, y=specificities, mode='lines', name='Specificity', line=dict(color = "#8B4513")))
fig.add_trace(go.Scatter(x=thresholds, y=f1_scores, mode='lines', name='F1 score', line = dict(color = 'rgb(44, 160, 44)')))


# Highlighting the best threshold
fig.add_shape(
    type="line",
    x0=thresholds[idx],
    x1=thresholds[idx],
    y0=0,
    y1=1,
    line=dict(color="grey", width=1, dash="dash"),
)

annotation_text = (f'Threshold={thresholds[idx]:.2f}<br>'
                   f'<span style="color:rgb(31, 119, 180)">Sensitivity={sensitivities[idx]:.2f}</span><br>'
                   f'<span style="color:#FA8072">Precision={precisions[idx]:.2f}</span><br>'
                   f'<span style="color:#8B4513">Specificity={specificities[idx]:.2f}</span><br>'
                   f'<span style="color:rgb(44, 160, 44)">F1 socre={f1_scores[idx]:.2f}</span>')


fig.add_annotation(
    x=thresholds[idx] + 0.01,  # move right by adding a value to x
    y=0.01,         # move down by subtracting a value from y
    align="left",
    xanchor="left",
    yanchor="bottom",
    text=annotation_text,
    showarrow=False,       # show the arrow
    arrowhead=2,          # style of the arrow
    axref="x",            # ensure the arrow points using x-coordinate
    ayref="y",            # ensure the arrow points using y-coordinate
    ax=thresholds[idx],       # x-coordinate of arrow's target
    ay=0.5,               # y-coordinate of arrow's target
    bgcolor="rgba(255, 255, 255, 0.5)",
    bordercolor="grey",
    borderwidth=1,
    borderpad=4
)


fig.update_layout(
    title="<b>Performance Metrics vs. Threshold <br>(CDAI non-remission with RAID)</b>",
    title_x=0.46,
    title_y=0.95,
    xaxis_title="Threshold",
    yaxis_title="Metrics",
    plot_bgcolor = 'white',
    title_font=dict(size=24, ),
    xaxis_title_font=dict(size=20, ),
    yaxis_title_font=dict(size=20, ),
    xaxis_tickfont=dict(size=14, ),
    yaxis_tickfont=dict(size=14, ),
    xaxis=dict(
        showgrid=False,
        tickformat = '.1f',
        zeroline=True,
        showline=True,
        mirror = True,
        linewidth=1,
        linecolor='black',
        ),
    yaxis=dict(showgrid=False,
               zeroline=True,
        showline=True,
               tickformat = '.1f',
        mirror = True,
        linewidth=1,
        linecolor='black',
               range=[0, 1]),
    width=800,
    height=600 )

fig.update_layout(
    legend=dict(
        font=dict(
            size=14
        )
    )
)

pio.renderers.default = 'colab'
fig.show()
print(f"Best Threshold: {thresholds[idx]}")

#calculate formula
t = logit_final.coef_
coef_flat = [item for sublist in t for item in sublist]
coef_flat

interce = logit_final.intercept_
interce

coeff_df = pd.DataFrame(coef_flat,X_ml.columns,columns=['Coefficient'])
coeff_df[coeff_df['Coefficient'] != 0]

value = X_ml['raid_tot_0'].iloc[4] * coef_flat[0] + interce[0]
1/(math.exp(-value)+1)

thre = 0.56 #this is based on obtaingin optimal sen/pre balance.

p = thre
raw_formula = 0 - math.log((1/p) - 1)
raid = (raw_formula - interce[0])/ coef_flat[0]
raid
