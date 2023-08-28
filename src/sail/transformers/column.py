from sklearn.base import OneToOneFeatureMixin

from .base import SAILTransformer


class ColumnNamePrefixTransformer(OneToOneFeatureMixin, SAILTransformer):
    def __init__(self, prefix="") -> None:
        super(ColumnNamePrefixTransformer, self).__init__()
        self.prefix = prefix
        self.validation_params["cast_to_ndarray"] = False

    def _partial_fit(self, X, y=None):
        return self

    def _transform(self, X, y=None, copy=None):
        if copy:
            new_df = X.copy()
        else:
            new_df = X

        new_columns = []
        for feature in X.columns:
            new_columns.append(self.prefix + "_" + feature)

        new_df.columns = new_columns
        self.feature_names_in_ = new_columns

        return new_df.to_numpy()
