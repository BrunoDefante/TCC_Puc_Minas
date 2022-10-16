from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class ProcessData(BaseEstimator, TransformerMixin):

    def __init__(self, trans_columns:dict, clean_inf:bool, clean_sup:bool, outliers_value) -> None:
        super().__init__()

        

        self.trans_columns = trans_columns
        self.outliers_value = outliers_value
        self.clean_inf = clean_inf
        self.clean_sup = clean_sup

    
    def find_outliers(self, X, y=None) -> tuple:
        if (isinstance(self.outliers_value, str))\
            and (self.outliers_value=='Tukey'):
            q1, q3 = pd.Series.quantile(X, [.25, .75])
            iqr = q3 - q1
            limit_sup = q3 + (1.5 * iqr)
            limit_inf = q1 - (1.5 * iqr)

            return (limit_inf, limit_sup)
    
    def fit(self, X, y=None, **fit_args):
        self.columns = [col for col in self.trans_columns.keys()]
        transfomed_columns = dict()

        for col in self.columns:
            f_limit_inf = self.trans_columns[col]['limit_inf']
            f_limit_sup = self.trans_columns[col]['limit_sup']
            x= X[col].copy()

            limit_inf, limit_sup = self.find_outliers(x)
            value_inf={'adj_inf_cp':None, 'adj_inf_cn':None}
            value_sup={'adj_sup_cp':None, 'adj_sup_cn':None}

            if f_limit_inf != None:
                adj_inf_cp = f_limit_inf(x.loc[(y==1) & (x>=limit_inf)])
                adj_inf_cn = f_limit_inf(x.loc[(y==0) & (x>=limit_inf)])

                value_inf={
                            'value_inf_cp':adj_inf_cp,
                            'value_inf_cn':adj_inf_cn,
                            'limit_inf':limit_inf
                        }
            if f_limit_sup != None:
                adj_sup_cp = f_limit_sup(x.loc[(y==1) & (x>=limit_sup)])
                adj_sup_cn = f_limit_sup(x.loc[(y==0) & (x>=limit_sup)])

                value_sup={
                            'value_sup_cp':adj_sup_cp,
                            'value_sup_cn':adj_sup_cn
                        }
            
            transfomed_columns[col] = {
                                        'value_inf':value_inf,
                                        'value_sup':value_sup,
                                        'limit_inf':limit_inf,
                                        'limit_sup':limit_sup
                                    }
        self.transfomed_columns = transfomed_columns


        return self

    def transform(self, X, y=None, **fit_params):
        for col in self.columns:
            tf_col = self.transfomed_columns[col]
            
            if self.clean_inf:
                mask = (X[col] < tf_col['limit_inf']) & (y==1)
                X.loc[mask , col] = tf_col['value_inf']['value_inf_cp']

                mask = (X[col] < tf_col['limit_inf']) & (y==0)
                X.loc[mask, col] = tf_col['value_inf']['value_inf_cn']

            if self.clean_sup:
                mask = (X[col] > tf_col['limit_sup']) & (y==1)
                X.loc[mask, col] = tf_col['value_sup']['value_sup_cp']
                
                mask = (X[col] > tf_col['limit_sup']) & (y==0)
                X.loc[mask, col] = tf_col['value_sup']['value_sup_cn']
                
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)