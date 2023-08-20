from river.compat import River2SKLTransformer
from sail.models.river.base import RiverMixin
from sklearn.utils._set_output import _get_output_config, _safe_set_output
from itertools import chain


class BaseRiverTransformer(River2SKLTransformer, RiverMixin):
    def __init__(self, *args, **Kwargs):
        super(BaseRiverTransformer, self).__init__(*args, **Kwargs)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        The feature names out will prefixed by the lowercased class name. For
        example, if the transformer outputs 3 features, then the feature names
        out are: `["class_name0", "class_name1", "class_name2"]`.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Only used to validate feature names with the names seen in :meth:`fit`.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        super().check_is_fitted(self, "_n_features_out")
        return super()._generate_get_feature_names_out(
            self, self._n_features_out, input_features=input_features
        )
