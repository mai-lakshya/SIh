import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from hybrid_model import TreeWrapperClassifier
import lightgbm as lgb

X = pd.DataFrame(np.random.rand(100, 5))
y = np.random.randint(0, 2, 100)

clf = TreeWrapperClassifier(lgb.LGBMClassifier, random_state=42, n_estimators=10)
print("Testing fit...")
clf.fit(X, y)
print("classes_ exists:", hasattr(clf, 'classes_'))

print("Testing cross_val_predict...")
cross_val_predict(clf, X, y, cv=2)
print("Done!")
