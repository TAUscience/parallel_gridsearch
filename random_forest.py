import itertools

# Hiperparametrso de Random Forest
hiper_grid_rf = {
    'n_estimators':  [100, 200, 300], 
    'criterion': ['gini','entropy','log_loss'],
    'min_samples_split': [2,3],
    'bootstrap': [True, False] 

}

keys_rf, values_rf = zip(*hiper_grid_rf.items())
combinations_rf = [dict(zip(keys_rf, v)) for v in itertools.product(*values_rf)]

def evaluate_rf(lock, hyperparameter_set, X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import recall_score

    for s in hyperparameter_set:
        clf=RandomForestClassifier()
        clf.set_params(n_estimators=s['n_estimators'], criterion=s['criterion'], min_samples_split =s['min_samples_split'],bootstrap=s['bootstrap'])
        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        lock.acquire()
        accuracy = accuracy_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        print(f"n_estimators: {s['n_estimators']}, "
          f"criterion: {s['criterion']}, "
          f"min_samples_split: {s['min_samples_split']}, "
          f"bootstrap: {s['bootstrap']}, acc: {accuracy:.4f}, recall: {recall:.4f}")
        lock.release()
