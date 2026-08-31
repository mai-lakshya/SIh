import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from hybrid_model import TreeWrapperClassifier
import lightgbm as lgb

X = pd.DataFrame(np.random.rand(100, 5))
y = np.random.randint(0, 2, 100)

clf = TreeWrapperClassifier(lgb.LGBMClassifier, random_state=42, n_estimators=10)
print("Testing cross_val_predict with predict_proba...")
cross_val_predict(clf, X, y, cv=2, method='predict_proba')
print("Done!")
