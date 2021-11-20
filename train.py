from sklearn.datasets import load_iris
from sklearn.svm import SVC
from joblib import dump

data = load_iris()
print(data['DESCR'])
X, y = data['data'], data['target']
clf = SVC()
clf.fit(X, y)
dump(clf, 'model.joblib')
