'''
Meta-estimators for building composite models with transformers
'''
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as Pipeline_imb

class PandasColumnTransformer(ColumnTransformer):
    def __init__(self, transformers, remainder_cols='drop', sparse_threshold=0.3, n_jobs=None, transformer_weights=None):
        """
        Implementation of sklearn ColumnTransformer that outputs a pandas.DataFrame instead of an array

        Parameters
        ----------
        
        transformers: list of tuples
            List of ``(name, transformer, columns)`` tuples specifying the transformer objects to be applied to subsets of the data.

        name: str
            Like in Pipeline and FeatureUnion, this allows the transformer and its parameters to be set using set_params and searched in grid search.
            
        transformer: {‘drop’, ‘passthrough’} or estimator
            Estimator must support fit and transform. Special-cased strings ‘drop’ and ‘passthrough’ are accepted as well, to indicate to drop the columns or to pass them through untransformed, respectively.
            
        columns: str, array-like of str, int, array-like of int, array-like of bool, slice or callable
            Indexes the data on its second axis. Integers are interpreted as positional columns, while strings can reference DataFrame columns by name. A scalar string or int should be used where transformer expects X to be a 1d array-like (vector), otherwise a 2d array will be passed to the transformer. A callable is passed the input data X and can return any of the above. To select multiple columns by name or dtype, you can use make_column_selector.

        remainder_cols: {‘drop’, ‘passthrough’} or estimator, default=’drop’
            By default, only the specified columns in transformers are transformed and combined in the output, and the non-specified columns are dropped. (default of 'drop'). By specifying remainder='passthrough', all remaining columns that were not specified in transformers will be automatically passed through. This subset of columns is concatenated with the output of the transformers. By setting remainder to be an estimator, the remaining non-specified columns will use the remainder estimator. The estimator must support fit and transform. Note that using this feature requires that the DataFrame columns input at fit and transform have identical order.

        sparse_threshold: float, default=0.3
            If the output of the different transformers contains sparse matrices, these will be stacked as a sparse matrix if the overall density is lower than this value. Use sparse_threshold=0 to always return dense. When the transformed output consists of all dense data, the stacked result will be dense, and this keyword will be ignored.

        n_jobs: int, default=None
            Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.

        transformer_weights: dict, default=None
            Multiplicative weights for features per transformer. The output of the transformer is multiplied by these weights. Keys are transformer names, values the weights.

        Attributes
        ----------        
        transformers_: list
            The collection of fitted transformers as tuples of (name, fitted_transformer, column). fitted_transformer can be an estimator, ‘drop’, or ‘passthrough’. In case there were no columns selected, this will be the unfitted transformer. If there are remaining columns, the final element is a tuple of the form: (‘remainder’, transformer, remaining_columns) corresponding to the remainder parameter. If there are remaining columns, then len(transformers_)==len(transformers)+1, otherwise len(transformers_)==len(transformers).

        named_transformers_: Bunch
            Access the fitted transformer by name.
        
        sparse_output_: bool
            Boolean flag indicating whether the output of transform is a sparse matrix or a dense numpy array, which depends on the output of the individual transformers and the sparse_threshold keyword.
        """
        super().__init__(transformers, remainder='drop', sparse_threshold=sparse_threshold, n_jobs=n_jobs, transformer_weights=transformer_weights)
        self.remainder_cols = remainder_cols

    
    def fit(self, X, y, **fit_params):
        """
        Fit all transformers using X.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data, of which specified subsets are used to fit the
            transformers.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        self : ColumnTransformer
            This estimator
        """
        foo = super().fit_transform(X, y)
        self.columns = X.columns

        return self

    def transform(self, X, y=None):
        """
        Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : pandas.DataFrame
            The data to be transformed by subset.

        Returns
        -------
        X_t : pandas.DataFrame
            Concatenation of results of transformers.
        """
        X_t = super().transform(X)

        if self.sparse_output_:
            X_t = X_t.todense()
        
        used_cols = []
        columns = []

        for _, trans, col, _ in self._iter(fitted=True):

            if trans=='drop' or trans=='passthrough':
                continue
            if type(col)==str:
                col = [col]
            used_cols += col

            if hasattr(trans, 'get_feature_names_out'):
                columns += list(trans.get_feature_names_out())
            else:
                columns += col

        df = pd.DataFrame(X_t, index=X.index, columns = columns)
        if self.remainder_cols == 'passthrough':
            df = pd.concat([df, X[[x for x in self.columns if x not in used_cols]]], axis=1)

        return df

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data, of which specified subsets are used to fit the
            transformers.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        X_t : pd.DataFrame
            Concatenation of results of transformers.
        """
        return self.fit(X, y).transform(X, y)