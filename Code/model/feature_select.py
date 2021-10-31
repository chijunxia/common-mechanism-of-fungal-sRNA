from AAAA_model_xin.feature_selector import FeatureSelector
import pandas as pd

train = pd.read_csv('D:\H\chi_paper_xin\Os14\Feature_select\Sample_label_feature_MFE_onehot_zscore.csv')
train_labels = train['class']
#print(train.head())
train = train.drop(train['class'])
fs = FeatureSelector(data = train, labels = train_labels)

#missing value
fs.identify_missing(missing_threshold=0.6)
missing_features = fs.ops['missing']
#print(missing_features[:10])
fs.plot_missing()
#print(fs.missing_stats.head(10))

#Single Unique Value
fs.identify_single_unique()
single_unique = fs.ops['single_unique']
#print(single_unique)
fs.plot_unique()
#print(fs.unique_stats.sample(5))

#Collinear (highly correlated) Features
fs.identify_collinear(correlation_threshold=0.98)
correlated_features = fs.ops['collinear']
#print(correlated_features[:5])
fs.plot_collinear()
fs.plot_collinear(plot_all=True)
#fs.identify_collinear(correlation_threshold=0.98)
#fs.plot_collinear()
#print(fs.record_collinear.head())

#Zero Importance Features
fs.identify_zero_importance(task = 'classification', eval_metric = 'auc', 
                            n_iterations = 10, early_stopping = True)
one_hot_features = fs.one_hot_features
base_features = fs.base_features
#print('There are %d original features' % len(base_features))
#print('There are %d one-hot features' % len(one_hot_features))
#print(fs.data_all.head(10))
zero_importance_features = fs.ops['zero_importance']
#print(zero_importance_features[10:15])
fs.plot_feature_importances(threshold = 0.99, plot_n = 12)
#print(fs.feature_importances.head(10))
one_hundred_features = list(fs.feature_importances.loc[:99, 'feature'])
#print(len(one_hundred_features))

#Low Importance Features
fs.identify_low_importance(cumulative_importance = 0.99)
low_importance_features = fs.ops['low_importance']
#print(low_importance_features[:5])

#Removing Features
#train_no_missing = fs.remove(methods = ['missing'])
#train_no_missing_zero = fs.remove(methods = ['missing', 'zero_importance'])
all_to_remove = fs.check_removal()
print(all_to_remove)
print(len(all_to_remove))
train_removed = fs.remove(methods = 'all')
print(train_removed)

