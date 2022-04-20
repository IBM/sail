# Mostly from: https://github.com/ylytkin/python-lasvm/blob/main/lasvm/lasvm.py

from typing import List, Union, Tuple, Optional

import numpy as np
from tqdm import tqdm

from lasvm.base_kernel_method import BaseKernelMethod
from sail.models.native.base import BaseEstimator

__all__ = [
    'LaSVM'
]


class LaSVM(BaseKernelMethod, BaseEstimator):
    """The LaSVM online learning algorithm for binary classification
    as described here:
    https://leon.bottou.org/papers/bordes-ertekin-weston-bottou-2005

    Note that some positive and negative samples are needed to
    initialize the model.

    Example
    -------
    x = np.array([[ 1.84, -1.7 ],
                  [-0.52,  0.27],
                  [-0.23, -0.26],
                  [-1.42,  0.17],
                  [ 1.  , -1.  ],
                  [ 0.01,  1.71],
                  [-0.53,  1.7 ],
                  [-0.27,  0.06]])

    y = np.array([1, 1, 0, 0, 1, 0, 0, 1])

    pos_samples = x[:2]
    neg_samples = x[2:4]

    lasvm = LaSVM(pos_samples, neg_samples)
    lasvm.fit(x, y)

    lasvm.score(x, y)  # 0.875

    Parameters
    ----------
    pos_samples : numpy array
        positive samples, shape (n_vectors, n_features)
    neg_samples : numpy array
        negative samples, shape (n_vectors, n_features)
    c : float, default 1
        regularization parameter
    kernel : {'rbf', 'linear', 'poly'}, default 'rbf'
        kernel name
    gamma : 'scale' or float, default 'scale'
        rbf and polynomial kernel gamma parameter (ignored for linear
        kernel). If 'scale' is passed, the actual value of gamma is
        calculated on initialization as::
            1 / x.shape[1] / x.var()
    degree : int, default 3
        polynomial kernel degree parameter (ignored for the rest of
        the kernels)
    coef0 : float, default 0
        polynomial kernel coef0 parameter (ignored for the rest of
        the kernels)
    tol : float
        tolerance parameter
    niter : int
        number of iterations on final step

    Attributes
    ----------
    c : float
        regularization parameter
    kernel : {'rbf', 'linear', 'poly'}
        kernel name
    gamma : 'scale' or float
        rbf and polynomial kernel gamma parameter
    gamma_ : float
        calculated gamma parameter
    degree : int, default 3
        polynomial kernel degree parameter
    coef0 : float, default 0
        polynomial kernel coef0 parameter
    tol : float
        tolerance parameter
    niter : int
        number of iterations on final step
    support_vectors : numpy array
        support vectors, shape (n_vectors, n_features)
    alpha : numpy array
        model coefficients, shape (n_vectors,)
    intercept : float
        model intercept
    target : numpy array
        support vector class labels, shape (n_vectors,)
    kernel_mx : numpy array
        pair-wise kernel values of support vectors,
        shape (n_vectors, n_vectors)
    gradient : numpy array
        gradient, shape (n_vectors,)
    a : numpy array
        lower bounds, shape (n_vectors,)
    b : numpy array
        upper bounds, shape (n_vectors,)
    delta : float
        current delta

    Methods
    -------
    partial_fit(x, y)
        partially fit the model on the training data
    finalize()
        finalize the training process after fitting is done
    fit(x, y)
        fit the model on the training data (equivalent to
        partial_fit with subsequent finalize)
    predict(x)
        predict class labels of the test data
    score(x, y)
        calculate model accuracy on the given data

    Properties
    ----------
    coef_ : numpy array
        coefficients of the separating hyperplane in the linear
        kernel case, shape (n_vectors,)
    """

    ERR = 0.00001

    class GradientPairNotFoundError(Exception):
        def __init__(self):
            super().__init__('could not find a maximum gradient pair')

    def __init__(
            self,
            pos_samples: np.ndarray,
            neg_samples: np.ndarray,
            c: float = 1,
            kernel: str = 'rbf',
            gamma: Union[float, str] = 'scale',
            degree: int = 3,
            coef0: float = 0,
            tol: float = 0.0001,
            niter: int = 10000,
    ) -> None:
        super().__init__(kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)

        self.c = c

        self.tol = tol

        self.target = np.empty(shape=(0,))
        self.kernel_mx = np.empty(shape=(0, 0))
        self.gradient = np.empty(shape=(0,))
        self.a = np.empty(shape=(0,))
        self.b = np.empty(shape=(0,))

        self.delta = None
        self.niter = niter

        self._initialize(pos_samples, neg_samples)

    def partial_fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            verbose: bool = False,
            shuffle: bool = False,
    ) -> 'LaSVM':
        """Partially (incrementally) fit model to the given data.

        Array `x` must be 2-dimensional of shape (n_vectors, n_features).
        Array `y` must be 1-dimensional of shape (n_vectors,) and contain only
        class labels 0 or 1.

        Parameters
        ----------
        x : numpy array
            data, shape (n_vectors, n_features)
        y : numpy array
            class labels, shape (n_vectors,), valued 0 or 1
        shuffle : bool
            shuffle data before fitting
        verbose : bool
            set verbosity

        Returns
        -------
        self
        """

        x = x.copy()
        y = self._prepare_targets(y)

        ids = np.arange(x.shape[0])

        if shuffle:
            np.random.shuffle(ids)

        for i in tqdm(ids, disable=not verbose):
            x0 = x[[i]]
            y0 = y[[i]]

            self._process(x=x0, y=y0)
            self._reprocess()

        return self

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            verbose: bool = False,
            shuffle: bool = False,
    ) -> 'LaSVM':
        """Fit model to the given data. Equivalent to partial_fit with
        subsequent finalize.

        Array `x` must be 2-dimensional of shape (n_vectors, n_features).
        Array `y` must be 1-dimensional of shape (n_vectors,) and contain only
        class labels 0 or 1.

        Parameters
        ----------
        x : numpy array
            data, shape (n_vectors, n_features)
        y : numpy array
            class labels, shape (n_vectors,), valued 0 or 1
        shuffle : bool
            shuffle data before fitting
        verbose : bool
            set verbosity

        Returns
        -------
        self
        """

        self.partial_fit(x, y, shuffle=shuffle, verbose=verbose)

        if verbose:
            print('Finalizing')

        self.finalize(verbose=verbose)

        return self

    def finalize(self, niter: Optional[int] = None, verbose: bool = False) -> 'LaSVM':
        """Perform the final step of learning (i.e. attempt to get delta
        below tau).

        Parameters
        ----------
        niter : int
            number of finalizing iterations
        verbose : bool
            set verbosity

        Returns
        -------
        self
        """

        niter = niter or self.niter

        tqdm_bar = tqdm(total=niter) if verbose else None
        broke_loop = False

        for _ in range(niter):
            self._reprocess()

            if self.delta <= self.tol:
                broke_loop = True
                break

            if verbose:
                tqdm_bar.update(1)

        if not broke_loop:
            print(f'Warning: delta did not converge below tol in {niter} iterations')

        if verbose:
            tqdm_bar.update(tqdm_bar.total - tqdm_bar.n)
            tqdm_bar.close()

        return self

    def __repr__(self):
        return 'LaSVM()'

    def _initialize(self, pos_samples: np.ndarray, neg_samples: np.ndarray) -> 'LaSVM':
        """Initialize model by adding some positive and negative samples as
        support vectors.

        Parameters
        ----------
        pos_samples : numpy array
            positive samples, shape (n_vectors, n_features)
        neg_samples : numpy array
            negative samples, shape (n_vectors, n_features)

        Returns
        -------
        self
        """

        self._remove_all_support_vectors()

        pos_samples = pos_samples.copy()
        neg_samples = neg_samples.copy()

        if self.gamma == 'scale':
            self.gamma_ = self._scaled_gamma(np.vstack([pos_samples, neg_samples]))

        self.support_vectors = np.empty(shape=(0, pos_samples.shape[1]))

        self._add_support_vectors(pos_samples, y=np.ones(pos_samples.shape[0]))
        self._add_support_vectors(neg_samples, y=- np.ones(neg_samples.shape[0]))

        i, j = self._find_maximum_gradient_pair()
        self.intercept = (self.gradient[i] + self.gradient[j]) / 2
        self.delta = self.gradient[i] - self.gradient[j]

        return self

    def _add_support_vectors(self, x: np.ndarray, y: np.ndarray) -> None:
        """Add support vectors with zero coefficients.

        Parameters
        ----------
        x : numpy array
            data, shape (n_vectors, n_features)
        y : numpy array
            class labels, shape (n_vectors, n_features)
        """

        n_vectors = x.shape[0]

        self.support_vectors = np.vstack([self.support_vectors, x])
        self.alpha = np.append(self.alpha, np.zeros(n_vectors))
        self.target = np.append(self.target, y)

        new_kernel_values = self._kernel(x, self.support_vectors)

        self.kernel_mx = np.vstack([self.kernel_mx, new_kernel_values[:, :-n_vectors]])
        self.kernel_mx = np.hstack([self.kernel_mx, new_kernel_values.T])

        gradient = y - new_kernel_values.dot(self.alpha)
        self.gradient = np.append(self.gradient, gradient)

        a = y * self.c
        print(type(a))
        a[a > 0] = 0
        self.a = np.append(self.a, a)

        b = y * self.c
        b[b < 0] = 0
        self.b = np.append(self.b, b)

    def _remove_support_vectors(self, vector_ids: List[int]) -> None:
        """Remove support vectors with given ids.

        Parameters
        ----------
        vector_ids : list
            ids of vectors to remove
        """

        self.support_vectors = np.delete(self.support_vectors, vector_ids, axis=0)
        self.alpha = np.delete(self.alpha, vector_ids)
        self.kernel_mx = np.delete(self.kernel_mx, vector_ids, axis=0)
        self.kernel_mx = np.delete(self.kernel_mx, vector_ids, axis=1)
        self.target = np.delete(self.target, vector_ids)
        self.gradient = np.delete(self.gradient, vector_ids)
        self.a = np.delete(self.a, vector_ids)
        self.b = np.delete(self.b, vector_ids)

    def _remove_all_support_vectors(self) -> None:
        n_sv = self.support_vectors.shape[0]

        if n_sv == 0:
            return

        to_remove = list(range(n_sv))
        self._remove_support_vectors(to_remove)

    def _is_violating_pair(self, i: int, j: int) -> bool:
        return self.alpha[i] < self.b[i] \
               and self.alpha[j] > self.a[j] \
               and self.gradient[i] - self.gradient[j] > self.tol

    def _find_max_gradient_id(self) -> int:
        """Find id of the vector with conditionally maximal gradient.

        Returns
        -------
        int
            vector id
        """

        mask = self.alpha < self.b
        mask_ids = np.where(mask)[0]
        i = mask_ids[np.argmax(self.gradient[mask])]

        return i

    def _find_min_gradient_id(self) -> int:
        """Find id of the vector with conditionally minimal gradient.

        Returns
        -------
        int
            vector id
        """

        mask = self.alpha > self.a
        mask_ids = np.where(mask)[0]
        j = mask_ids[np.argmin(self.gradient[mask])]

        return j

    def _find_maximum_gradient_pair(self) -> Tuple[int, int]:
        return self._find_max_gradient_id(), self._find_min_gradient_id()

    def _update_parameters(self, i: int, j: int) -> None:
        lambda_ = min(
            (self.gradient[i] - self.gradient[j]) / (
                        self.kernel_mx[i, i] + self.kernel_mx[j, j] - 2 * self.kernel_mx[i, j]),
            self.b[i] - self.alpha[i],
            self.alpha[j] - self.a[j],
        )

        self.alpha[i] = self.alpha[i] + lambda_
        self.alpha[j] = self.alpha[j] - lambda_

        self.gradient = self.gradient - lambda_ * (self.kernel_mx[i] - self.kernel_mx[j])

    def _process(self, x: np.ndarray, y: np.ndarray) -> None:
        """Process an object-target pair.

        Parameters
        ----------
        x : numpy array
            feature vector, shape (1, n_features)
        y : numpy array
            class label, shape (1,)
        """

        if (((self.support_vectors - x) ** 2).sum(axis=1) ** 0.5 < self.ERR).any():
            return

        self._add_support_vectors(x, y)

        if y[0] == 1:
            i = self.support_vectors.shape[0] - 1
            j = self._find_min_gradient_id()

        else:
            j = self.support_vectors.shape[0] - 1
            i = self._find_max_gradient_id()

        if not self._is_violating_pair(i, j):
            return

        self._update_parameters(i=i, j=j)

    def _reprocess(self) -> None:
        """Reprocess.
        """

        i, j = self._find_maximum_gradient_pair()

        if not self._is_violating_pair(i, j):
            return

        self._update_parameters(i=i, j=j)

        i, j = self._find_maximum_gradient_pair()
        to_remove = []

        for k in np.where(np.abs(self.alpha) < self.ERR)[0]:
            if (self.target[k] == -1 and self.gradient[k] >= self.gradient[i]) \
                    or (self.target[k] == 1 and self.gradient[k] <= self.gradient[j]):
                to_remove.append(k)

        self.intercept = (self.gradient[i] + self.gradient[j]) / 2
        self.delta = self.gradient[i] - self.gradient[j]

        self._remove_support_vectors(to_remove)

