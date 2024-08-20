
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

def get_batch_correlation(batch: np.ndarray)->np.float_:

    pearson_coefs = np.corrcoef(batch)
    n,_ = batch.shape

    iu1 = np.triu_indices(n=n,k=1)

    return np.mean(pearson_coefs[iu1])

def passive_sampling(X,y,seed,k,X_holdout,y_holdout):
    random_test_mse = []
    correlations = []
    num_features = X.shape[1]
    #split 20% of the data for training a model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=seed)
    
    model = RandomForestRegressor()
    # now select one sample randomly to add to the population and remove from test
    #do this for up to 50% of the data as training
    while len(X_train)<len(X)/2:
        X_batch = np.zeros((0,num_features))
        for i in range(k):
            random_index = np.random.choice(X_test.index)
            random_X = X_test.loc[random_index]
            random_y = y_test.loc[random_index]

            X_batch = np.append(X_batch,random_X.values.reshape(1,-1),axis=0)
            
            X_train = X_train.append(random_X)
            y_train = y_train.append(pd.Series(random_y))

            X_test = X_test.drop(random_index)
            y_test = y_test.drop(random_index)

        model.fit(X_train, y_train)
    
        corr = get_batch_correlation(X_batch)
        correlations.append(corr)

        #get the test accuracy by testing model on test data
        y_pred = model.predict(X_holdout)
        mse_pred = mean_squared_error(y_holdout, y_pred)
        random_test_mse.append(mse_pred)

    return random_test_mse, correlations



def active_sampling(X,y,seed,k,X_holdout,y_holdout):
    active_test_mse = []
    correlations = []
    num_features = X.shape[1]
    #split 20% of the data for training a model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=seed)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    #get the test accuracy by testing model on test data
    y_pred = model.predict(X_holdout)
    mse_pred = mean_squared_error(y_holdout, y_pred)
    # now select one sample randomly to add to the population and remove from test
    #do this for up to 50% of the data as training
    while len(X_train)<len(X)/2:
        X_batch = np.zeros((0,num_features))
        #uncertainty sampling

        #make predictions using each individual tree
        individual_tree_predictions = []
        for tree in model.estimators_:
            tree_prediction = tree.predict(X_test)
            individual_tree_predictions.append(tree_prediction)

        #compute the variance of predictions across trees
        variances = np.var(individual_tree_predictions, axis=0)
        variances = pd.Series(variances, index=X_test.index)
        #get the max variance index
        max_var_indices = variances.nlargest(k).index
        for max_var_index in max_var_indices:
            add_X = X_test.loc[max_var_index]
            add_y = y_test.loc[max_var_index]

            X_batch = np.append(X_batch,add_X.values.reshape(1,-1),axis=0)

            X_train = X_train.append(add_X)
            y_train = y_train.append(pd.Series(add_y))

        #remove selected indices from test set
        X_test = X_test.drop(max_var_indices)
        y_test = y_test.drop(max_var_indices)

        model.fit(X_train, y_train)

        corr = get_batch_correlation(X_batch)
        correlations.append(corr)

        #get the test accuracy by testing model on test data
        y_pred = model.predict(X_holdout)
        mse_pred = mean_squared_error(y_holdout, y_pred)
        active_test_mse.append(mse_pred)

    return active_test_mse, correlations


def hierarchical_sampling(X,y,seed,k,X_holdout,y_holdout):
    active_test_mse = []
    correlations = []
    num_features = X.shape[1]

    #split 20% of the data for training a model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=seed)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    # now select one sample randomly to add to the population and remove from test
    #do this for up to 50% of the data as training
    while len(X_train)<len(X)/2:
        X_batch = np.zeros((0,num_features))
        X_test.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        individual_tree_predictions = []
        for tree in model.estimators_:
            tree_prediction = tree.predict(X_test)
            individual_tree_predictions.append(tree_prediction)

        #compute the variance of predictions across trees
        variances = np.var(individual_tree_predictions, axis=0)
        #convert the list of variances to a pandas Series if needed
        variances = pd.Series(variances, index=X_test.index)

        #perform hierarchical clustering of the test data
        clustering = AgglomerativeClustering(n_clusters=k).fit(X_test)
        cluster_labels = clustering.labels_
        #loop through the clusters
        for cluster_label in range(k):
            max_variance = -1
            max_variance_index = -1
            #get the data within that cluster number
            cluster_indices = np.where(cluster_labels == cluster_label)[0]
    
            for index in cluster_indices:
                #determine which index has max variance
                variance = variances[index]
                if variance > max_variance:
                    max_variance = variance
                    max_variance_index = index
            if max_variance_index==-1:
                max_variance_index=cluster_indices[0]
            #add the most uncertain point from that cluster to the training data
            add_X = X_test.loc[max_variance_index]
            add_y = y_test.loc[max_variance_index]
            X_batch = np.append(X_batch,add_X.values.reshape(1,-1),axis=0)

            X_train = X_train.append(add_X)
            y_train = y_train.append(pd.Series(add_y))

            X_test = X_test.drop(max_variance_index)
            y_test = y_test.drop(max_variance_index)
        
        model.fit(X_train, y_train)

        corr = get_batch_correlation(X_batch)
        correlations.append(corr)

        #get the test accuracy by testing model on test data
        y_pred = model.predict(X_holdout)
        mse_pred = mean_squared_error(y_holdout, y_pred)
        active_test_mse.append(mse_pred)

    return active_test_mse, correlations


def kmeans_sampling(X, y, seed, k,X_holdout,y_holdout):
    active_test_mse = []
    correlations = []
    num_features = X.shape[1]

    #split 20% of the data for training a model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=seed)
    #n_estimators=50, max_depth=20, max_features='sqrt', n_jobs=-1
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    #now select one sample randomly to add to the population and remove from test
    #do this for up to 50% of the data as training
    while len(X_train) < len(X) / 2:
        X_batch = np.zeros((0,num_features))
        #reindex X_test from 0
        X_test.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        individual_tree_predictions = []
        for tree in model.estimators_:
            tree_prediction = tree.predict(X_test)
            individual_tree_predictions.append(tree_prediction)

        #compute the variance of predictions across trees
        variances = np.var(individual_tree_predictions, axis=0)
        #convert the list of variances to a pandas Series if needed
        variances = pd.Series(variances, index=X_test.index)

        #perform KMeans clustering of the test data
        clustering = KMeans(n_clusters=k, random_state=seed).fit(X_test)
        cluster_labels = clustering.labels_

        #loop through the clusters
        for cluster_label in range(k):
            max_variance = -1
            max_variance_index = -1
            #get the data within that cluster number
            cluster_indices = np.where(cluster_labels == cluster_label)[0]
            for index in cluster_indices:
                #determine which index has max variance
                variance = variances[index]
                if variance > max_variance:
                    max_variance = variance
                    max_variance_index = index
            if max_variance_index==-1:
                max_variance_index=cluster_indices[0]
            #add the most uncertain point from that cluster to the training data
            add_X = X_test.loc[max_variance_index]
            add_y = y_test.loc[max_variance_index]
            X_batch = np.append(X_batch,add_X.values.reshape(1,-1),axis=0)

            X_train = X_train.append(add_X)
            y_train = y_train.append(pd.Series(add_y))

            X_test = X_test.drop(max_variance_index)
            y_test = y_test.drop(max_variance_index)
            
        model.fit(X_train, y_train)

        corr = get_batch_correlation(X_batch)
        correlations.append(corr)

        #get the test accuracy by testing model on test data
        y_pred = model.predict(X_holdout)
        mse_pred = mean_squared_error(y_holdout, y_pred)
        active_test_mse.append(mse_pred)

    return active_test_mse, correlations


def run_simulations(k,X,y,dataset_name):

    #random seeds for splitting data
    seeds = [123, 456, 89, 5, 17]

    random_train_mse = []
    random_corr = []
    active_train_mse = []
    active_corr = []
    h_train_mse = []
    h_corr = []
    k_train_mse = []
    k_corr = []


    #loop through simulations
    for seed in seeds:
        print('SEED',seed)
        # PASSIVE
        #set random seed
        np.random.seed(seed)
        X_holdout, X_train, y_holdout, y_train = train_test_split(X, y, test_size=0.8, random_state=seed)
        print('PASSIVE')
        random_train_mse_seed, random_corr_seed = passive_sampling(X_train,y_train,seed,k,X_holdout,y_holdout)
        #print(random_train_mse_seed)
        
        random_train_mse.append(random_train_mse_seed)
        random_corr.append(random_corr_seed)


        # ACTIVE UNCERTAINTY SAMPLING
        print('ACITVE')
        active_train_mse_seed, active_corr_seed = active_sampling(X_train,y_train,seed,k,X_holdout,y_holdout)
        active_train_mse.append(active_train_mse_seed)
        active_corr.append(active_corr_seed)

        #HIERARCHICAL UNCERTAINTY SAMPLING
        print('HIERARCHICAL')
        h_train_mse_seed , h_corr_seed= hierarchical_sampling(X_train,y_train,seed,k,X_holdout,y_holdout)
        h_train_mse.append(h_train_mse_seed)
        h_corr.append(h_corr_seed)

        #K-MEANS UNCERTAINTY SAMPLING
        print('K-MEANS')
        k_train_mse_seed , k_corr_seed= kmeans_sampling(X_train,y_train,seed,k,X_holdout,y_holdout)
        k_train_mse.append(k_train_mse_seed)
        k_corr.append(k_corr_seed)

    # save the output results to numpy files
    k_train_mse = np.array(k_train_mse)
    h_train_mse = np.array(h_train_mse)
    active_train_mse = np.array(active_train_mse)
    random_train_mse = np.array(random_train_mse)
    random_corr = np.array(random_corr)
    active_corr = np.array(active_corr)
    h_corr = np.array(h_corr)
    k_corr = np.array(k_corr)

    filename_format = "{}_randomforest_{}_{}.npy"
    filename_format_corr = "{}_randomforest_{}_{}_cor.npy"

    k_filename = filename_format.format(dataset_name, 'k-means', k)
    np.save(k_filename,k_train_mse)

    h_filename = filename_format.format(dataset_name, 'hierarchical', k)
    np.save(h_filename,h_train_mse)

    a_filename = filename_format.format(dataset_name, 'UC', k)
    np.save(a_filename,active_train_mse)

    r_filename = filename_format.format(dataset_name, 'random', k)
    np.save(r_filename,random_train_mse)

    r_filename = filename_format_corr.format(dataset_name,'random',k)
    np.save(r_filename,random_corr)

    filename = filename_format_corr.format(dataset_name,'k-means',k)
    np.save(filename,k_corr)

    filename = filename_format_corr.format(dataset_name,'hierarchical',k)
    np.save(filename,h_corr)

    filename = filename_format_corr.format(dataset_name,'UC',k)
    np.save(filename,active_corr)


data = pd.read_csv('../data/Inhibition_data/inhibition_features.csv')
y = data['% Control']
X = data.drop(columns=['Unnamed: 0','Protein HMS LINCS ID','% Control'])

k=4
dataset_name = 'Inhibition_data'
run_simulations(k,X,y,dataset_name)

k=8
run_simulations(k,X,y,dataset_name)

X = np.load('../data/abalone_age/X.npy')
X = pd.DataFrame(X)
y = np.load('../data/abalone_age/y.npy')
y = pd.DataFrame(y)
y = y[0]


k=16
dataset_name = 'abalone_age'
run_simulations(k,X,y,dataset_name)

k=32
run_simulations(k,X,y,dataset_name)

X = np.load('../data/logd74/fingerprints.npy')
X = pd.DataFrame(X)
y = np.load('../data/logd74/logd74s.npy')
y = pd.DataFrame(y)
y = y[0]

k=16
dataset_name = 'logd74'
run_simulations(k,X,y,dataset_name)

k=32
run_simulations(k,X,y,dataset_name)



