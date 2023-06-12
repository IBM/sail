import os
import warnings

import ray.cloudpickle as cpickle
from tune_sklearn._trainable import _Trainable


class _TuneSklearnTrainable(_Trainable):
    def save_checkpoint(self, checkpoint_dir):
        """Creates a checkpoint in ``checkpoint_dir``, creating a pickle file.

        Args:
            checkpoint_dir (str): file path to store pickle checkpoint.

        Returns:
            path (str): file path to the pickled checkpoint file.

        """
        # fit main estimator using the trial parameters.
        self.main_estimator.fit(self.X, self.y, **self.fit_params)

        path = os.path.join(checkpoint_dir, "sail_pipeline.pickle")
        try:
            with open(path, "wb") as f:
                cpickle.dump(self.main_estimator, f)
        except Exception:
            warnings.warn("Unable to save estimator.", category=RuntimeWarning)
        return path
