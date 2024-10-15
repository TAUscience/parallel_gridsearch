import itertools

# Parameters
param_grid_knn = {
    "n_neighbors" : [5, 10, 13, 15],
    "weights" : ["uniform", "distance"],
    "algorithm" : ["ball_tree", "kd_tree"],
    "p" : [1, 2] 
}

# Combinations
keys, values = zip(*param_grid_knn.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Function
def evaluate_knn(lock, hyperparameter_set, X_train, X_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import recall_score

    for subset in hyperparameter_set:
        clf = KNeighborsClassifier()
        clf.set_params(n_neighbors = subset["n_neighbors"], 
                       weights = subset["weights"],
                       algorithm = subset["algorithm"],
                       p = subset["p"])
        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        
        lock.acquire()
        accuracy = accuracy_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        print(f"n_neighbors: {subset['n_neighbors']}, weights: {subset['weights']}, algorithm: {subset['algorithm']}, p: {subset['p']}, acc: {accuracy}, recall: {recall}")
        lock.release()





