# project intro
"""
The goal of this project is to learn from home depot product search relevance data and
create a model to predict/score the relevance of pairs of search queries and products
to behave similar to human scorers
"""
#-------------------------------
# imports
#-------------------------------
import numpy as np
import pylab as pl
import pandas as pd
import nltk
import os
import collections
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn import metrics
from sklearn import cross_validation
from sklearn import grid_search
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer

#-------------------------------
# global variables
#-------------------------------
full_data_file = "full_data"
preprocessed_data_file = "preprocessed_data.csv"
df_all = pd.DataFrame()
num_train = 74067
num_test = 166693
stemmer = SnowballStemmer('english')


#-------------------------------
# function definitions
#-------------------------------
def str_stemmer(s):
  global stemmer
  return " ".join([stemmer.stem(word) for word in s.lower().split()])
def num_common_words(str1, str2):
  return sum(int(str2.find(word)>=0) for word in str1.split())
def num_common_words_type(str1, str2, type): # returns number of common words of a specific POS
  type_list1 = [x for (x,y) in nltk.pos_tag(nltk.word_tokenize(str1)) if y == type]
  return sum(int(str2.find(word)>=0) for word in type_list1)

def maybe_load_all_data(force=False):
  global df_all, num_train, num_test
  if force or not os.path.exists(full_data_file):
    # read and pre-process input data
    print "Reading input files..."
    df_train = pd.read_csv('./input/train.csv', encoding="ISO-8859-1")
    df_test = pd.read_csv('./input/test.csv', encoding="ISO-8859-1")
    df_pro_desc = pd.read_csv('./input/product_descriptions.csv')
    df_attr = pd.read_csv('./input/attributes.csv')

    print "unique attributes: {}".format(df_attr.name.nunique())

    # print high-level stats about data
    num_train = df_train.shape[0]
    num_test = df_test.shape[0]
    print "num_train: {}".format(num_train)
    print "num_test: {}".format(num_test)

    # pre-process input data
    print "Pre-processing data:"
    print "Concatenating train and test data..."
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    # print "df_all after concat: \n{}".format(df_all.head(5))


    # pre-process attributes
    df_attr.dropna(how="all", inplace=True)
    df_attr["product_uid"] = df_attr["product_uid"].astype(int)
    df_attr["value"] = df_attr["value"].astype(str)
    def concate_attrs(attrs):
        """
        attrs is all attributes of the same product_uid
        """
        names = attrs["name"]
        values = attrs["value"]
        pairs  = []
        for n, v in zip(names, values):
            pairs.append(' '.join((n, v)))
        return ' '.join(pairs)

    df_pro_attrs = df_attr.groupby("product_uid").apply(concate_attrs)
    df_pro_attrs = df_pro_attrs.reset_index(name="product_attributes")

    # Merging data
    print "Merging with product description..."
    df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
    # print "df_all after merge: \n{}".format(df_all.head(5))

    print "Merging with product attributes..."
    df_all = pd.merge(df_all, df_pro_attrs, how="left", on="product_uid")
    df_all['product_attributes'] = df_all['product_attributes'].fillna('')


    # stemming all text data
    print "Stemming text columns..."
    df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
    df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
    df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))
    df_all['product_attributes'] = df_all['product_attributes'].map(lambda x:str_stemmer(x.decode('utf-8')))
    df_all.to_pickle(full_data_file)
  else:
    "Reading full data from file: " + full_data_file
    df_all = pd.read_pickle('./' + full_data_file)


def maybe_preprocess_all_data(force=False):
  global df_all, num_train, num_test
  if force or not os.path.exists(preprocessed_data_file):
    # Creating additional input fields
    print "Add new input fields..."
    df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
    df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']+"\t"+df_all['product_attributes']
    df_all['num_word_in_title'] = df_all['product_info'].map(lambda x:num_common_words(x.split('\t')[0],x.split('\t')[1]))
    df_all['num_word_in_description'] = df_all['product_info'].map(lambda x:num_common_words(x.split('\t')[0],x.split('\t')[2]))
    df_all['num_word_in_attributes'] = df_all['product_info'].map(lambda x:num_common_words(x.split('\t')[0],x.split('\t')[3]))

    print "adding JJ input fields..."
    df_all['num_JJ_word_in_title'] = df_all['product_info'].map(lambda x:num_common_words_type(x.split('\t')[0],x.split('\t')[1], "JJ"))
    df_all['num_JJ_word_in_description'] = df_all['product_info'].map(lambda x:num_common_words_type(x.split('\t')[0],x.split('\t')[2], "JJ"))
    df_all['num_JJ_word_in_attributes'] = df_all['product_info'].map(lambda x:num_common_words_type(x.split('\t')[0],x.split('\t')[3], "JJ"))

    print "adding NN input fields..."
    df_all['num_NN_word_in_title'] = df_all['product_info'].map(lambda x:num_common_words_type(x.split('\t')[0],x.split('\t')[1], "NN"))
    df_all['num_NN_word_in_description'] = df_all['product_info'].map(lambda x:num_common_words_type(x.split('\t')[0],x.split('\t')[2], "NN"))
    df_all['num_NN_word_in_attributes'] = df_all['product_info'].map(lambda x:num_common_words_type(x.split('\t')[0],x.split('\t')[3], "NN"))

    # Dropping unnecessary input fields
    # print "df_all before dropping: {}".format(df_all.head(5))
    df_all = df_all.drop(['search_term','product_title','product_description','product_info', 'product_attributes'],axis=1)
    df_all.to_pickle(preprocessed_data_file)
  else:
    "Reading preprocessed data from file: " + preprocessed_data_file
    df_all = pd.read_csv('./' + preprocessed_data_file)

# make sure output is in range 1-3
def make_in_bounds(array):
    output = np.asarray(array)
    low_val = 1
    high_val = 3
    output[output < low_val] = low_val
    output[output > high_val] = high_val
    return output.tolist()

#-------------------------------
# Main Script
#-------------------------------

# maybe_load_all_data()
maybe_preprocess_all_data()

# randomly run split data into train and validation sets
df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']

y_train = df_train['relevance'].values
x_train = df_train.drop(['id','relevance'],axis=1)
x_test = df_test.drop(['id','relevance'],axis=1).values

validation_set_ratio = 0.5
x_t, x_v, y_t, y_v = cross_validation.train_test_split(x_train, y_train, test_size=validation_set_ratio, random_state=0)
# kf = cross_validation.KFold(num_train, n_folds=3, shuffle=True, random_state=0)
# skf = cross_validation.StratifiedKFold(y_train, n_folds=3, shuffle=True, random_state=0)

#-------------------------------
# exploratory data analysis
#-------------------------------

# stats on y_train
print "Minimum y: {}".format(np.min(y_train))
print "Maximum y: {}".format(np.max(y_train))
print "Mean of y: {}".format(np.mean(y_train))
print "STD of y: {}".format(np.std(y_train))

y_train_distict = sorted(list(set(y_train)))
print "Set of outputs: {}".format(collections.Counter(y_train))

# stats on x_train
print "Pre-processed dataset has shape: {}".format(x_train.shape)
print x_train.head()
print "\n\nInput correlation matrix: available in correlation_matrix.csv"

print "\n\Mean for each input feature:"
print x_train.mean()
print "\n\nVariance for each input feature:"
print x_train.std()

# printing sorted list of correlated features
corr = pd.DataFrame(x_train.corr())
cols = np.array(corr.columns)
n = corr.shape[0]
list = []
for i in range(n):
    for j in range(i+1,n):
        list.append( (corr.as_matrix()[i,j],cols[i]+ " & " + cols[j]) )
for i,j in sorted(list):
    print i,j

# PCA analysis
from sklearn.decomposition import PCA
pca = PCA(n_components = 6)
pca.fit(x_train)

# Print the components and the amount of variance in the data contained in each dimension
print "PCA Components:"
print pca.components_

print "\nPCA Explained Variance Ratio:"
print pca.explained_variance_ratio_

# Assigning a label (1, 2, 3) to train data and visualizing the labels
def labeler(x):
    if x <= 1.5:
        return 1
    if x < 2.5:
        return 2
    else:
        return 3
y_train_labels = map(labeler, y_train)

# Mapping data to first two PCA's and visualizing output labels based on proximity to values 1, 2, or 3
pca = PCA(n_components=2)
X_r = pca.fit(x_train).transform(x_train)

print np.min(X_r[:,0])
print np.max(X_r[:,0])
print np.min(X_r[:,1])
print np.max(X_r[:,1])
print np.mean(X_r[:,0])
print np.mean(X_r[:,1])

y = np.array(y_train_labels)
target_names = [1,2,3]

# plt.figure()

# plt.subplot(3, 1, 1)
# plt.scatter(X_r[y == 1, 0], X_r[y == 1, 1], c="r", label="Label 1", alpha = 0.2, s=1)
# plt.legend()
# plt.subplot(3, 1, 2)
# plt.scatter(X_r[y == 2, 0], X_r[y == 2, 1], c="g", label="Label 2", alpha = 0.2, s=1)
# plt.legend()
# plt.subplot(3, 1, 3)
# plt.scatter(X_r[y == 3, 0], X_r[y == 3, 1], c="b", label="Label 3", alpha = 0.2, s=1)
# plt.legend()

# # for c, i, target_name in zip("rgb", [1, 2, 3], target_names):
# #     plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name, alpha = 0.4, s=1)
# # plt.legend()
# plt.title('PCA of dataset')
# plt.show()



# print "\nThe figure below shows total explained variance of data over the number of PC's"
# x = np.arange(6)
# import matplotlib.pyplot as plt

# print np.cumsum(pca.explained_variance_ratio_) #
# plt.title('Total Explained Variance vs. Number of PC\'s')
# plt.plot(x+1, np.cumsum(pca.explained_variance_ratio_), '-')
# # plt.axis([1, 6, 0.0, 1.000])
# plt.show()


# analyze Linear Regressor performance
clf = LinearRegression()
print "Learning train data using Linear Regressor.."
clf.fit(x_t, y_t)
print "LinearRegressor MSE score: {}".format(mean_squared_error(y_v, clf.predict(x_v)))

# analyze KNN Regressor performance
clf = KNeighborsRegressor()
print "Learning train data using KNeighborsRegressor.."
clf.fit(x_t, y_t)
print "KNeighborsRegressor MSE score: {}".format(mean_squared_error(y_v, clf.predict(x_v)))

# analyze DecisionTree regressor performance
clf = DecisionTreeRegressor()
print "Learning train data using DecisionTree.."
clf.fit(x_t, y_t)
print "DecisionTree MSE: {}".format(mean_squared_error(y_v, clf.predict(x_v)))

# analyze PLS regressor performance
from sklearn.cross_decomposition import PLSRegression
clf = PLSRegression(n_components=2)
print "Learning train data using PLSRegressor.."
clf.fit(x_t, y_t)
print "PLSRegressor MSE: {}".format(mean_squared_error(y_v, clf.predict(x_v)))

# analyze AdaBoostRegressor performance
clf = AdaBoostRegressor()
print "Learning train data using AdaBoostRegressor.."
clf.fit(x_t, y_t)
print "AdaBoostRegressor MSE: {}".format(mean_squared_error(y_v, clf.predict(x_v)))

# analyze RandomForest regressor performance
rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
print "Learning train data using RandomForestRegressor.."
clf.fit(x_t, y_t)
print "RandomForestRegressor score: {}".format(mean_squared_error(y_v, clf.predict(x_v)))

# choose the best regressor and create final report
rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
y_pred = make_in_bounds(y_pred)
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)


