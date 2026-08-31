import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from hybrid_model import TreeWrapperClassifier
import lightgbm as lgb

X = pd.DataFrame(np.random.rand(100, 5))
y = np.random.randint(0, 2, 100)

clf = TreeWrapperClassifier(lgb.LGBMClassifier, random_state=42, n_estimators=10)
stack = StackingClassifier(estimators=[('lgb', clf)], final_estimator=LogisticRegression(), cv=2)

print("Testing StackingClassifier.fit()...")
stack.fit(X, y)
print("Done!")
