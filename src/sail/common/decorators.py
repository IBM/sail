from sklearn.utils import check_array, check_X_y


def log_epoch(func):
    def wrapper(*args, **kwargs):
        estimator = args[0]
        estimator.verbosity.log_epoch(args[1])
        func(*args, **kwargs)

    return wrapper


def validate_X_y(func):
    def wrapper(*args, **kwargs):
        X = args[1]
        y = args[2]

        fit_params = {}
        if len(args) == 4:
            fit_params = args[3]

        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))

        X_new = X.copy()
        datetime_cols = list(X_new.select_dtypes(include="datetime64[ns]"))
        X_new = X_new.drop(datetime_cols, axis=1)

        if y is None:
            _ = check_array(
                X_new,
                dtype=None,
                input_name="X",
            )
        else:
            _, y = check_X_y(X_new, y, dtype=None)

        func(args[0], X=X, y=y, **fit_params)

    return wrapper
