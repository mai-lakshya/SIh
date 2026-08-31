import re

with open('c:/Users/26beevlsi049/.gemini/antigravity-ide/scratch/sih-personal-2-main/hybrid_model.py', 'r') as f:
    content = f.read()

# Add RidgeClassifierCV
content = content.replace(
    "from sklearn.linear_model import LogisticRegressionCV, RidgeCV",
    "from sklearn.linear_model import LogisticRegressionCV, RidgeCV, RidgeClassifierCV"
)

# Insert TreeWrapper classes after safe_logit
tree_wrapper_code = """
class TreeWrapperBase(BaseEstimator):
    def __init__(self, model_class, random_state=42, **kwargs):
        self.model_class = model_class
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y):
        X_np = X.values if hasattr(X, 'values') else X
        y_np = np.array(y)
        
        stratify = y_np if hasattr(self, 'classes_') else None
        if len(X_np) < 20:
            X_tr, X_val, y_tr, y_val = X_np, X_np, y_np, y_np
        else:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_np, y_np, test_size=0.1, random_state=self.random_state, stratify=stratify
            )
            
        # extract early_stopping_rounds
        esr = self.kwargs.pop('early_stopping_rounds', None)
        
        # recreate model with remaining kwargs
        self.model = self.model_class(random_state=self.random_state, **self.kwargs)
        
        fit_kwargs = {}
        if esr is not None:
            if 'XGB' in self.model_class.__name__:
                self.model.set_params(early_stopping_rounds=esr)
                fit_kwargs['eval_set'] = [(X_val, y_val)]
                fit_kwargs['verbose'] = False
            elif 'LGBM' in self.model_class.__name__:
                import lightgbm as lgb
                fit_kwargs['eval_set'] = [(X_val, y_val)]
                fit_kwargs['callbacks'] = [lgb.early_stopping(stopping_rounds=esr, verbose=False)]
            elif 'CatBoost' in self.model_class.__name__:
                self.model.set_params(early_stopping_rounds=esr)
                fit_kwargs['eval_set'] = [(X_val, y_val)]
                fit_kwargs['verbose'] = False
                
        self.model.fit(X_tr, y_tr, **fit_kwargs)
        
        if hasattr(self, 'classes_'):
            self.classes_ = np.unique(y_np)
        return self

    def predict(self, X):
        X_np = X.values if hasattr(X, 'values') else X
        return self.model.predict(X_np)

class TreeWrapperClassifier(TreeWrapperBase, ClassifierMixin):
    def __init__(self, model_class, random_state=42, **kwargs):
        super().__init__(model_class, random_state, **kwargs)
        
    def predict_proba(self, X):
        X_np = X.values if hasattr(X, 'values') else X
        return self.model.predict_proba(X_np)

class TreeWrapperRegressor(TreeWrapperBase, RegressorMixin):
    def __init__(self, model_class, random_state=42, **kwargs):
        super().__init__(model_class, random_state, **kwargs)

"""

content = content.replace("class TabNetWrapperClassifier", tree_wrapper_code + "class TabNetWrapperClassifier")

# FTTransformer Early Stopping
ftt_orig = """class FTTransformerWrapperBase(BaseEstimator):
    def __init__(self, is_classifier=True, d_token=32, n_blocks=3, epochs=500, batch_size=128, lr=1e-3, random_state=42):
        self.is_classifier = is_classifier
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, X, y):
        X_np = X.values if hasattr(X, 'values') else X
        X_np = X_np.astype(np.float32)
        y_np = np.array(y)
        
        if self.is_classifier:
            self.classes_ = np.unique(y_np)
            y_np = y_np.astype(np.float32).reshape(-1, 1)
            out_dim = 1
        else:
            y_np = y_np.astype(np.float32).reshape(-1, 1)
            out_dim = 1
            
        self.model = FTTransformerMLP(n_features=X_np.shape[1], d_token=self.d_token, n_blocks=self.n_blocks, out_dim=out_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss() if self.is_classifier else nn.MSELoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        dataset = TensorDataset(torch.tensor(X_np), torch.tensor(y_np))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            self.model.train()
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()
        return self"""

ftt_new = """class FTTransformerWrapperBase(BaseEstimator):
    def __init__(self, is_classifier=True, d_token=32, n_blocks=3, epochs=500, batch_size=128, lr=1e-3, random_state=42, patience=30):
        self.is_classifier = is_classifier
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.patience = patience
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, X, y):
        X_np = X.values if hasattr(X, 'values') else X
        X_np = X_np.astype(np.float32)
        y_np = np.array(y)
        
        stratify = y_np if self.is_classifier else None
        if len(X_np) < 20:
             X_tr, X_val, y_tr, y_val = X_np, X_np, y_np, y_np
        else:
             X_tr, X_val, y_tr, y_val = train_test_split(X_np, y_np, test_size=0.1, random_state=self.random_state, stratify=stratify)
             
        if self.is_classifier:
            self.classes_ = np.unique(y_np)
            y_tr = y_tr.astype(np.float32).reshape(-1, 1)
            y_val = y_val.astype(np.float32).reshape(-1, 1)
            out_dim = 1
        else:
            y_tr = y_tr.astype(np.float32).reshape(-1, 1)
            y_val = y_val.astype(np.float32).reshape(-1, 1)
            out_dim = 1
            
        self.model = FTTransformerMLP(n_features=X_np.shape[1], d_token=self.d_token, n_blocks=self.n_blocks, out_dim=out_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss() if self.is_classifier else nn.MSELoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        dataset = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        val_x = torch.tensor(X_val).to(self.device)
        val_y = torch.tensor(y_val).to(self.device)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(val_x)
                val_loss = criterion(val_preds, val_y).item()
                
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                break
        return self"""

content = content.replace(ftt_orig, ftt_new)

# Update _build_classifiers
clf_orig = """        try:
            lgb_clf = lgb.LGBMClassifier(random_state=self.random_state, **lgb_params)
        except Exception:
            lgb_clf = lgb.LGBMClassifier(random_state=self.random_state)
            
        xgb_clf = xgb.XGBClassifier(random_state=self.random_state, **xgb_params)
        cat_clf = CatBoostClassifier(verbose=0, random_state=self.random_state, **cat_params)
        tab_clf = TabNetWrapperClassifier(random_state=self.random_state, **tab_params)
        ft_clf = FTTransformerWrapperClassifier(random_state=self.random_state, **ftt_params)"""

clf_new = """        lgb_clf = TreeWrapperClassifier(lgb.LGBMClassifier, random_state=self.random_state, **lgb_params)
        xgb_clf = TreeWrapperClassifier(xgb.XGBClassifier, random_state=self.random_state, **xgb_params)
        cat_clf = TreeWrapperClassifier(CatBoostClassifier, random_state=self.random_state, **cat_params)
        tab_clf = TabNetWrapperClassifier(random_state=self.random_state, **tab_params)
        ft_clf = FTTransformerWrapperClassifier(random_state=self.random_state, **ftt_params)"""

content = content.replace(clf_orig, clf_new)

# Update LogisticRegressionCV to RidgeClassifierCV
meta_clf_orig = """        meta_classifier = Pipeline([
            ('logit', FunctionTransformer(safe_logit)),
            ('lr', LogisticRegressionCV(Cs=np.logspace(-4, 1, 20), cv=10, penalty='elasticnet', solver='saga', l1_ratios=np.linspace(0.1, 1.0, 10), random_state=self.random_state, max_iter=2000))
        ])"""

meta_clf_new = """        meta_classifier = Pipeline([
            ('logit', FunctionTransformer(safe_logit)),
            ('lr', RidgeClassifierCV(alphas=np.logspace(-3, 4, 30), cv=10))
        ])"""

content = content.replace(meta_clf_orig, meta_clf_new)

# Update _build_regressors
reg_orig = """        xgb_reg = xgb.XGBRegressor(random_state=self.random_state, **xgb_params)
        cat_reg = CatBoostRegressor(verbose=0, random_state=self.random_state, **cat_params)
        try:
            lgb_reg = lgb.LGBMRegressor(random_state=self.random_state, **lgb_params)
        except Exception:
            lgb_reg = lgb.LGBMRegressor(random_state=self.random_state)"""

reg_new = """        xgb_reg = TreeWrapperRegressor(xgb.XGBRegressor, random_state=self.random_state, **xgb_params)
        cat_reg = TreeWrapperRegressor(CatBoostRegressor, random_state=self.random_state, **cat_params)
        lgb_reg = TreeWrapperRegressor(lgb.LGBMRegressor, random_state=self.random_state, **lgb_params)"""

content = content.replace(reg_orig, reg_new)

with open('c:/Users/26beevlsi049/.gemini/antigravity-ide/scratch/sih-personal-2-main/hybrid_model.py', 'w') as f:
    f.write(content)
