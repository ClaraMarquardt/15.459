# -------------------------------------------------
# Initialization ----------------------------------
# -------------------------------------------------

## Folders

### Define folders
app_dir  = "/Users/claramarquardt/Google_Drive/University & School/MIT/Academic/Courses/Fall Semester (2019-2020)/Classes/Financial data science/15.459/AssignmentD/python/"
dir_data = "/Users/claramarquardt/Google_Drive/University & School/MIT/Academic/Courses/Fall Semester (2019-2020)/Classes/Financial data science/Assignment D/data/"

### Append code folder
import sys
import os
sys.path.append(os.path.normpath(app_dir))

# ------

## Libraries

from dependencies import *

# ------

## Dependency tests

### Load functions
sys.path.append(os.path.normpath(app_dir))
from dependency_test import *

### Test spacy and nltk
nltk_test()
spacy_test()

# -------------------------------------------------
# Data --------------------------------------------
# -------------------------------------------------

## Connect to the database
# * Server:obelix.mit.edu | Windows authentication | username: student | password: equity

db_conn = pyodbc.connect('Driver={/usr/local/lib/libmsodbcsql.13.dylib};'
                         'Server=obelix.mit.edu;'
                         'Database=rcv1;'
                         'uid=student;'
                         'pwd=equity;')

# ------

## Extract, perform basic cleaning, and save the tables 

###  h1
db_table_h1 = pd.read_sql('SELECT * FROM h1', db_conn)
db_table_h1.to_pickle(dir_data+"db_table_h1.pkl")
print(len(db_table_h1))

###  h2
db_table_h2 = pd.read_sql('SELECT * FROM h2', db_conn)
db_table_h2.to_pickle(dir_data+"db_table_h2.pkl")
print(len(db_table_h2))

###  h3
db_table_h3 = pd.read_sql('SELECT * FROM h3', db_conn)
db_table_h3.to_pickle(dir_data+"db_table_h3.pkl")
print(len(db_table_h3))

###  h4
db_table_h4 = pd.read_sql('SELECT * FROM h4', db_conn)
db_table_h4.to_pickle(dir_data+"db_table_h4.pkl")
print(len(db_table_h4))

###  h5
db_table_h5 = pd.read_sql('SELECT * FROM h5', db_conn)
db_table_h5.to_pickle(dir_data+"db_table_h5.pkl")
print(len(db_table_h5))

### news & news_train
db_table_news_train         = pd.read_sql('SELECT * FROM news', db_conn)
db_table_news_train['test'] = 0
print(len(db_table_news_train))

db_table_news_test          = pd.read_sql('SELECT * FROM news_test', db_conn)
db_table_news_test['test']  = 1
print(len(db_table_news_test))

db_table_news               = db_table_news_train.append(db_table_news_test)
db_table_news.to_pickle(dir_data+"db_table_news.pkl")
print(len(db_table_news))

### news_topics & news_topics 
db_table_news_topics        = pd.read_sql('SELECT * FROM news_topics', db_conn)
db_table_news_topics["cat"] = [x.strip() for x in db_table_news_topics["cat"]]

db_table_news_topics.to_pickle(dir_data+"db_table_news_topics.pkl")
print(len(db_table_news_topics))

db_table_news_topics_old    = pd.read_sql('SELECT * FROM news_topics_old', db_conn)
db_table_news_topics_old.to_pickle(dir_data+"db_table_news_topics_old.pkl")
print(len(db_table_news_topics_old))

### topics & topics_orig
db_table_topics             = pd.read_sql('SELECT * FROM topics', db_conn)
db_table_topics.to_pickle(dir_data+"db_table_topics.pkl")
print(len(db_table_topics))

db_table_topics_orig        = pd.read_sql('SELECT * FROM topics_orig', db_conn)
db_table_topics_orig.to_pickle(dir_data+"db_table_topics_orig.pkl")
print(len(db_table_topics_orig))

### * Read in the tables
# db_table_h1              = pd.read_pickle(dir_data+"db_table_h1.pkl")
# db_table_h2              = pd.read_pickle(dir_data+"db_table_h2.pkl")
# db_table_h3              = pd.read_pickle(dir_data+"db_table_h3.pkl")
# db_table_h4              = pd.read_pickle(dir_data+"db_table_h4.pkl")
# db_table_h5              = pd.read_pickle(dir_data+"db_table_h5.pkl")
# db_table_news            = pd.read_pickle(dir_data+"db_table_news.pkl")
# db_table_news_topics     = pd.read_pickle(dir_data+"db_table_news_topics.pkl")
# db_table_news_topics_old = pd.read_pickle(dir_data+"db_table_news_topics_old.pkl")
# db_table_topics          = pd.read_pickle(dir_data+"db_table_topics.pkl")
# db_table_topics_orig     = pd.read_pickle(dir_data+"db_table_topics_orig.pkl")

# -------------------------------------------------
# Exploration -------------------------------------
# -------------------------------------------------

# ------

## data and topic structure exploration

### number of articles & number of articles with labels
len(set(db_table_news_topics["id"])) # 804,414 articles with labels
len(set(db_table_news["id"]))        # 413,765 articles in joint testing and training set
len(db_table_news["id"])             # 413,765 unique articles in joint testing and training set

len(set(set(db_table_news_topics["id"]) - set(db_table_news["id"]))) # 390,649 articles with labels for which do not have underlying data
len(set(set(db_table_news["id"]) - set(db_table_news_topics["id"]))) # 0 articles in joint testing and training set do not have a label

### * identify subset of news topics labels that correspond to data in joing testing and training set (i.e., labels for 413,765 articles)
db_table_news_topics["testtrain"] = 0
db_table_news_topics.loc[db_table_news_topics["id"].isin(db_table_news["id"]),"testtrain"]=1
len(set(db_table_news_topics[db_table_news_topics["testtrain"]==1]["id"]))

### number of topics
len(set(db_table_news_topics["cat"]))  # 103 unique topic labels
len(set(db_table_news_topics[db_table_news_topics["testtrain"]==1]["cat"]))  # 103 unique topic labels associated with data in joint testing and training set

### * merge in topic levels
db_table_news_topics.loc[db_table_news_topics["cat"].isin(db_table_h1["h1"]),"catlevel"]=int(1)
db_table_news_topics.loc[db_table_news_topics["cat"].isin(db_table_h2["h2"]),"catlevel"]=int(2)
db_table_news_topics.loc[db_table_news_topics["cat"].isin(db_table_h3["h3"]),"catlevel"]=int(3)
db_table_news_topics.loc[db_table_news_topics["cat"].isin(db_table_h4["h4"]),"catlevel"]=int(4)
db_table_news_topics.loc[db_table_news_topics["cat"].isin(db_table_h5["h5"]),"catlevel"]=int(5)
db_table_news_topics["catlevel"] = db_table_news_topics["catlevel"].astype(int)
db_table_news_topics["cat"] = [x.strip() for x in db_table_news_topics["cat"]]

### distribution of topics
pd.crosstab(index=[
	db_table_news_topics[db_table_news_topics["testtrain"]==1]["cat"],
	db_table_news_topics[db_table_news_topics["testtrain"]==1]["catlevel"]], 
	columns="count", normalize="columns").sort_values('count', ascending=False)*100

pd.crosstab(index=[
	db_table_news_topics[(db_table_news_topics["testtrain"]==1) & (db_table_news_topics["catlevel"]==1)]["cat"],
	db_table_news_topics[(db_table_news_topics["testtrain"]==1) & (db_table_news_topics["catlevel"]==1)]["catlevel"]], 
	columns="count", normalize="columns").sort_values('count', ascending=False)*100
pd.crosstab(index=[
	db_table_news_topics[(db_table_news_topics["testtrain"]==1) & (db_table_news_topics["catlevel"]==2)]["cat"],
	db_table_news_topics[(db_table_news_topics["testtrain"]==1) & (db_table_news_topics["catlevel"]==2)]["catlevel"]], 
	columns="count", normalize="columns").sort_values('count', ascending=False)*100
pd.crosstab(index=[
	db_table_news_topics[(db_table_news_topics["testtrain"]==1) & (db_table_news_topics["catlevel"]==3)]["cat"],
	db_table_news_topics[(db_table_news_topics["testtrain"]==1) & (db_table_news_topics["catlevel"]==3)]["catlevel"]], 
	columns="count", normalize="columns").sort_values('count', ascending=False)*100
pd.crosstab(index=[
	db_table_news_topics[(db_table_news_topics["testtrain"]==1) & (db_table_news_topics["catlevel"]==4)]["cat"],
	db_table_news_topics[(db_table_news_topics["testtrain"]==1) & (db_table_news_topics["catlevel"]==4)]["catlevel"]], 
	columns="count", normalize="columns").sort_values('count', ascending=False)*100
pd.crosstab(index=[
	db_table_news_topics[(db_table_news_topics["testtrain"]==1) & (db_table_news_topics["catlevel"]==5)]["cat"],
	db_table_news_topics[(db_table_news_topics["testtrain"]==1) & (db_table_news_topics["catlevel"]==5)]["catlevel"]], 
	columns="count", normalize="columns").sort_values('count', ascending=False)*100

### distribution of the number of topics per article
db_table_news_topics[db_table_news_topics["testtrain"]==1].groupby("id").count().sort_values('cat', ascending=False)

label_count = db_table_news_topics[db_table_news_topics["testtrain"]==1].groupby("id").count().sort_values('cat', ascending=False)
label_count[label_count["cat"]==1]
label_count[label_count["cat"]==2]
label_count[label_count["cat"]>2]

db_table_news_topics[(db_table_news_topics["testtrain"]==1) & (db_table_news_topics["catlevel"]==1)].groupby("id").count().sort_values('cat', ascending=False)["cat"]
db_table_news_topics[(db_table_news_topics["testtrain"]==1) & (db_table_news_topics["catlevel"]==2)].groupby("id").count().sort_values('cat', ascending=False)["cat"]
db_table_news_topics[(db_table_news_topics["testtrain"]==1) & (db_table_news_topics["catlevel"]==3)].groupby("id").count().sort_values('cat', ascending=False)["cat"]
db_table_news_topics[(db_table_news_topics["testtrain"]==1) & (db_table_news_topics["catlevel"]==4)].groupby("id").count().sort_values('cat', ascending=False)["cat"]
db_table_news_topics[(db_table_news_topics["testtrain"]==1) & (db_table_news_topics["catlevel"]==5)].groupby("id").count().sort_values('cat', ascending=False)["cat"]

# db_table_news_topics[db_table_news_topics["id"]==2290]

# -------------------------------------------------
# Analysis ----------------------------------------
# -------------------------------------------------

# ------

## parameters

### data set
test_length    = len(db_table_news[db_table_news["test"]==1])
train_length   = len(db_table_news[db_table_news["test"]==0])

### features
min_df_value   = 0.03

### training & co
cv_fold        = 5

# ------

## identify and potentially subset testing and training data
news_id_test  = np.array(db_table_news[db_table_news["test"]==1]["id"])
news_id_train = np.array(db_table_news[db_table_news["test"]==1]["id"])[0:train_length]

# ------

## create features for joint test and training set

### * create frequency–inverse document frequency (tf-idf) table
news_tfidf_vectorizer      = TfidfVectorizer(use_idf=True, min_df=min_df_value)
news_tfidf_vectors_train   = news_tfidf_vectorizer.fit_transform(db_table_news.loc[db_table_news["id"].isin(news_id_train),"article"])
news_tfidf_vectors_test    = news_tfidf_vectorizer.transform(db_table_news.loc[db_table_news["id"].isin(news_id_test),"article"])

### examine tf-idf table
# news_tfidf_vectors_train.get_shape()
# news_tfidf_vectors_test.get_shape()
# df = pd.DataFrame(news_tfidf_vectors_train[0].T.todense(), index=news_tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
# df.sort_values(by=["tfidf"],ascending=False)

## create outputs for joint test and training set

### * one hot encode categories
db_table_news_topics_cat         = pd.Series(db_table_news_topics["cat"])
db_table_news_topics_cat_dummies = pd.get_dummies(db_table_news_topics_cat)
db_table_news_topics             = pd.concat([db_table_news_topics, db_table_news_topics_cat_dummies], axis=1, join='inner')

db_table_news_topics_1 = pd.concat(
	[pd.DataFrame(db_table_news_topics.groupby(['id'])['CCAT'].agg('sum')), 
	pd.DataFrame(db_table_news_topics.groupby(['id'])['ECAT'].agg('sum')), 
	pd.DataFrame(db_table_news_topics.groupby(['id'])['GCAT'].agg('sum')),
	pd.DataFrame(db_table_news_topics.groupby(['id'])['MCAT'].agg('sum'))], axis=1, join='inner')
db_table_news_topics_1['id'] = db_table_news_topics_1.index

### * create top level output table
topic_1 = []
for CCAT, ECAT, GCAT, MCAT in zip(
	db_table_news_topics_1['CCAT'], 
	db_table_news_topics_1['ECAT'], 
	db_table_news_topics_1['GCAT'],
	db_table_news_topics_1['MCAT']):
		topic_1.append([CCAT,ECAT,GCAT,MCAT])

# ------

## top-level classifiers - one vs. rest (one classifier per class)

### define variables
x_train = news_tfidf_vectors_train
x_test  = news_tfidf_vectors_test

y_train = np.array(topic_1)[np.array(db_table_news_topics_1['id'].isin(news_id_train)) ]
y_test  = np.array(topic_1)[np.array(db_table_news_topics_1['id'].isin(news_id_test)) ]


### define models

#### > support vector classification (LogisticRegression)
model_lg           = OneVsRestClassifier(LogisticRegression())
param_grid_lg      = {'estimator__C': [0.001,0.01,0.1,1,10,100]}  
param_grid_lg_add  = {}


#### > support vector classification (LinearSVC)
model_svm           = OneVsRestClassifier(LinearSVC())
param_grid_svm      = {'estimator__C': [0.001, 0.1, 0.5, 1, 1.5, 5, 10, 100, 1000]}  
param_grid_svm_add  = {}

#### > random forest (RandomForestClassifier)
model_rf          = OneVsRestClassifier(RandomForestClassifier())
param_grid_rf     = {'estimator__n_estimators': [250],
 				     'estimator__max_features': ['sqrt'],
                     'estimator__min_samples_split': [6]}
param_grid_rf_add = {}

#### > gradient boosted trees (XGB)
model_xgb           = OneVsRestClassifier(XGBClassifier())
param_grid_xgb      = {'estimator__max_depth':[3,5],
                       'estimator__min_child_weight':range(1,6,3), 
                       'estimator__gamma':[i/10.0 for i in range(0,5)]}
param_grid_xgb_add  = {'estimator__learning_rate':[0.01,0.1,1]}


#### > model list & parameter list
model_name_list     = ["Logistic Regression(LogisticRegression)", "SVM (LinearSVC)", "RF (Random Forest)", "XGB (Gradient Boosted Trees)"]
model_list          = [model_lg, model_svm, model_rf, model_xgb]
parameter_list      = [param_grid_lg, param_grid_svm, param_grid_rf, param_grid_xgb]
parameter_list_add  = [param_grid_lg_add, param_grid_svm_add, param_grid_rf_add, param_grid_xgb_add]

### modelling pipeline
result_metric     = dict()
model_tuned_list  = dict()

for (model_name_x, model_x,param_grid_x, param_grid_add_x) in zip(model_name_list, model_list, parameter_list, parameter_list_add):
	
	print(model_name_x)

	# fit model
	model      = model_x.fit(x_train, y_train)

	# tune model
	grid = GridSearchCV(model_x, 
		param_grid_x, 
		refit = True, 
		verbose = 3,
		cv = cv_fold, 
		scoring = 'accuracy') 
	grid.fit(x_train, y_train)

	if (len(param_grid_add_x)>0):
		grid = GridSearchCV(grid.best_estimator_, 
			param_grid_add_x, 
			refit = True, 
			verbose = 3,
			cv = cv_fold, 
			scoring = 'accuracy') 
		grid.fit(x_train, y_train)

	# obtain tuned model
	model_tuned      = grid.best_estimator_
	pred_train_tuned = model_tuned.predict(x_train) 
	
	# obtain predictions
	pred_train       = model.predict(x_train) 
	pred_train_tuned = model_tuned.predict(x_train) 
	pred_test        = model.predict(x_test) 
	pred_test_tuned  = model_tuned.predict(x_test) 

	# assess model

	## in-sample

	### > accuracy - subset accuracy 
	accuracy_score_in_sample       = accuracy_score(y_train,pred_train)
	accuracy_score_in_sample_tuned = accuracy_score(y_train,pred_train_tuned)
	# sum([all(y_train[x]==pred_train[x]) for x in range(0,train_length)])/train_length

	### > accuracy - hamming score/loss
	hamming_score_in_sample       = 1 - hamming_loss(y_train,pred_train)
	hamming_score_in_sample_tuned = 1 - hamming_loss(y_train,pred_train_tuned)
	# np.mean([sum(y_train[x]==pred_train[x])/4 for x in range(0,train_length)])

	## cross validate

	### > accuracy - subset accuracy 
	accuracy_score_CV       = np.mean(cross_val_score(model, x_train, y_train, cv=cv_fold, scoring='accuracy'))
	accuracy_score_CV_tuned = np.mean(cross_val_score(model_tuned, x_train, y_train, cv=cv_fold, scoring='accuracy'))

	### > accuracy - hamming score/loss
	hamming_score_CV        = np.mean(1-(cross_val_score(model, x_train, y_train, cv=cv_fold, scoring=make_scorer(hamming_loss,greater_is_better=False)) * -1))
	hamming_score_CV_tuned  = np.mean(1-(cross_val_score(model_tuned, x_train, y_train, cv=cv_fold, scoring=make_scorer(hamming_loss,greater_is_better=False))*-1))

	## out-of-sample

	### > accuracy - hamming score/loss
	accuracy_score_out_of_sample        = accuracy_score(y_test,pred_test)
	accuracy_score_out_of_sample_tuned  = accuracy_score(y_test,pred_test_tuned)
	# sum([all(y_test[x]==pred_test[x]) for x in range(0,test_length)])/test_length

	### > accuracy - hamming score/loss
	hamming_score_out_of_sample         = 1 - hamming_loss(y_test,pred_test)
	hamming_score_out_of_sample_tuned   = 1 - hamming_loss(y_test,pred_test_tuned)
	# np.mean([sum(y_test[x]==pred_test[x])/4 for x in range(0,test_length)])

	result_metric[model_name_x] = dict()
	result_metric[model_name_x]["accuracy_score_in_sample"]           = accuracy_score_in_sample
	result_metric[model_name_x]["accuracy_score_in_sample_tuned"]     = accuracy_score_in_sample_tuned
	result_metric[model_name_x]["accuracy_score_CV"]                  = accuracy_score_CV
	result_metric[model_name_x]["accuracy_score_CV_tuned"]            = accuracy_score_CV_tuned
	result_metric[model_name_x]["accuracy_score_CV_tuned_check"]      = grid.best_score_
	result_metric[model_name_x]["accuracy_score_out_of_sample"]       = accuracy_score_out_of_sample
	result_metric[model_name_x]["accuracy_score_out_of_sample_tuned"] = accuracy_score_out_of_sample_tuned

	result_metric[model_name_x]["hamming_score_in_sample"]            = hamming_score_in_sample
	result_metric[model_name_x]["hamming_score_in_sample_tuned"]      = hamming_score_in_sample_tuned
	result_metric[model_name_x]["hamming_score_CV"]                   = hamming_score_CV
	result_metric[model_name_x]["hamming_score_CV_tuned"]             = hamming_score_CV_tuned
	result_metric[model_name_x]["hamming_score_out_of_sample"]        = hamming_score_out_of_sample
	result_metric[model_name_x]["hamming_score_out_of_sample_tuned"]  = hamming_score_out_of_sample_tuned

	model_tuned_list[model_name_x]                                    = model_tuned
	
	print("Finished: " + model_name_x)
	print(list(param_grid_x.keys()))
	print([model.get_params()[list(param_grid_x.keys())[x]] for x in range(0, len(list(param_grid_x.keys())))])
	print([model_tuned.get_params()[list(param_grid_x.keys())[x]] for x in range(0, len(list(param_grid_x.keys())))])
	if (len(param_grid_add_x)>0):
		print(list(param_grid_add_x.keys()))
		print([model.get_params()[list(param_grid_add_x.keys())[x]] for x in range(0, len(list(param_grid_add_x.keys())))])
		print([model_tuned.get_params()[list(param_grid_add_x.keys())[x]] for x in range(0, len(list(param_grid_add_x.keys())))])

	print(result_metric[model_name_x])

	f = open(dir_data+"result_metric.pkl","wb")
	pickle.dump(result_metric,f)
	f.close()

	f = open(dir_data+"model_tuned_list.pkl","wb")
	pickle.dump(model_tuned_list,f)
	f.close()

### assessment pipeline (external dataset)

#### > select optimal model
model_test_ext = model_tuned_list["LG (Logistic Regression)"]
# model_test_ext = model_tuned_list["SVM (LinearSVC)"]
# model_test_ext = model_tuned_list["RF (Random Forest)"]
# model_test_ext = model_tuned_list["XGB (Gradient Boosted Trees)"]

#### > load the data
db_table_news_ext            = pd.read_csv(dir_data+"db_table_news_ext.csv")

#### > create features 
news_tfidf_vectors_test_ext  = news_tfidf_vectorizer.transform(db_table_news_ext["article"])

#### > create predictions
pred_test_ext                = pd.DataFrame(model_test_ext.predict(news_tfidf_vectors_test_ext))
pred_test_ext.columns        = ["CCAT_pred","ECAT_pred","GCAT_pred","MCAT_pred"]

#### > save dataset
db_table_news_ext  = pd.concat([db_table_news_ext, pred_test_ext], axis=1, join='inner')
db_table_news_ext  = db_table_news_ext[["id","CCAT_pred","ECAT_pred","GCAT_pred","MCAT_pred"]]
db_table_news_ext.to_csv(dir_data+"db_table_news_ext_annotated.csv")

