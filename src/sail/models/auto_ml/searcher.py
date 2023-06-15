from typing import Dict, List, Optional
from tune_sklearn.list_searcher import ListSearcher, RandomListSearcher
from ray.tune.utils.util import _atomic_save, _load_newest_checkpoint


class SearcherSerializationMixin:
    def get_state(self):
        state = self.__dict__.copy()
        return state

    def set_state(self, state):
        self.__dict__.update(state)

    def save_to_dir(self, dirpath, session_str):
        state_dict = self.get_state()
        _atomic_save(
            state=state_dict,
            checkpoint_dir=dirpath,
            file_name=self.CKPT_FILE_TMPL.format(session_str),
            tmp_file_name=".tmp_generator",
        )

    def restore_from_dir(self, dirpath: str):
        """Restores self + searcher + search wrappers from dirpath."""
        state_dict = _load_newest_checkpoint(dirpath, self.CKPT_FILE_TMPL.format("*"))
        if not state_dict:
            raise RuntimeError("Unable to find checkpoint in {}.".format(dirpath))
        self.set_state(state_dict)


class SailListSearcher(SearcherSerializationMixin, ListSearcher):
    """Custom search algorithm to support passing in a list of
    dictionaries to SailTuneGridSearchCV

    """

    CKPT_FILE_TMPL = "sail-list-searcher-state-{}.json"

    def __init__(self, param_grid, points_to_evaluate: Optional[List[Dict]] = None):
        super(SailListSearcher, self).__init__(param_grid)
        self._points_to_evaluate = points_to_evaluate
        self._setup()

    def _setup(self) -> None:
        if self._points_to_evaluate is None:
            self._best_configurations = []
        else:
            self._best_configurations = self._points_to_evaluate

    def suggest(self, trial_id):
        if self._best_configurations:
            return self._best_configurations.pop(0)
        else:
            return super().suggest(trial_id)


class SailRandomListSearcher(SearcherSerializationMixin, RandomListSearcher):
    """Custom search algorithm to support passing in a list of
    dictionaries to SailTuneSearchCV

    """

    CKPT_FILE_TMPL = "random-list-searcher-state-{}.json"

    def __init__(self, param_grid, points_to_evaluate: Optional[List[Dict]] = None):
        super(SailRandomListSearcher, self).__init__(param_grid)
        self._points_to_evaluate = points_to_evaluate
        self._setup()

    def _setup(self) -> None:
        if self._points_to_evaluate is None:
            self._best_configurations = []
        else:
            self._best_configurations = self._points_to_evaluate

    def suggest(self, trial_id):
        if self._best_configurations:
            return self._best_configurations.pop(0)
        else:
            return super().suggest(trial_id)
