"""
Weighted Gradient learning based LSTM (WGLSTM) neural network

Code adapted from https://github.com/weilai0980/onlineLearning
"""

import copy
from types import SimpleNamespace

import numpy as np
import scipy.stats as st
import tensorflow as tf
from numpy.lib.scimath import sqrt
from scikeras.wrappers import KerasRegressor

from sail.models.keras.base import KerasSerializationMixin
from sail.utils.stats import nmse


class _Model(tf.keras.models.Sequential):
    def __init__(
        self,
        num_of_features=1,
        hidden_layer_neurons=450,
        hidden_layer_activation="linear",
        regularization_factor=0.0001,
        timesteps=1,
        window_size=20,
    ):
        super(_Model, self).__init__(name="WGLSTM")
        self.window_size = window_size
        self.add(
            tf.keras.layers.LSTM(
                hidden_layer_neurons,
                return_sequences=True,
                stateful=True,
                batch_input_shape=(1, timesteps, num_of_features),
                kernel_regularizer=tf.keras.regularizers.l2(
                    regularization_factor
                ),
            )
        )
        self.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=num_of_features)
            )
        )
        self.add(tf.keras.layers.Activation(hidden_layer_activation))

    def fit(self, x, y, **kwargs):
        """
        Trains the model for a fixed number of epochs
        (iterations on a dataset).
        """

        x = x.reshape((1, x.shape[0], 1))
        y = y.reshape((1, y.shape[0], 1))

        y_orig = []
        y_pred = []
        cur_pred = 0.0
        ts_timestamp = []
        outlier_flag = [0] * (x.shape[1] + 1)
        conf_level = 0.005
        win_susp = [0] * self.window_size
        resi_mean = x[:, 0:1, :][0][0][0]
        resi_sqr = x[:, 0:1, :][0][0][0] ** 2
        resi_var = 1.0
        seg_cnt = 1
        pre_susp_bool = False
        sus_point_bool = False

        for i in range(x.shape[1] - 1):
            cur_trnx = x[:, i : i + 1, :]
            cur_trny = y[:, i : i + 1, :]

            tmp_cur_trnx = copy.deepcopy(x[:, i : i + 1, :])
            tmp_cur_trny = copy.deepcopy(y[:, i : i + 1, :])

            weight = 1.0
            sus_point_bool = False

            if pre_susp_bool is True:
                tmp_cur_trnx[0][0][0] = cur_pred

            if i < self.window_size + 1:
                self.train_on_batch(cur_trnx, cur_trny)
                continue
            else:
                cur_prediction = self.predict_on_batch(tmp_cur_trnx)
                cur_pred_ini = cur_prediction[0][0][0]
                tmpresi = tmp_cur_trny[0][0][0] - cur_pred_ini

                (
                    norm_diff,
                    susp_diff,
                    point_diff,
                ) = self.sliding_window_features(i + 1, x, y, win_susp)

                tmpw = self.sliding_window_weight(
                    norm_diff,
                    susp_diff,
                    point_diff,
                )

                if len(norm_diff) == 0:
                    norm_diff = [0.0]

                testval = point_diff * 1.0 / (sum(norm_diff) / len(norm_diff))
                curmean = resi_mean
                curvar = resi_var
                tmp_zval = (tmpresi - curmean) * 1.0 / sqrt(curvar)
                tmp_pro_conve = st.norm.cdf(tmp_zval)

                if (
                    tmp_pro_conve > (1 - conf_level)
                    or tmp_pro_conve < conf_level
                ):
                    sus_point_bool = True
                    pre_susp_bool = True
                    weight = tmpw
                    win_susp.append(1)
                    win_susp.pop(0)

                    if outlier_flag[i + 1] == 1:
                        print(
                            "outlier weight at ",
                            i + 1,
                            ": YES ",
                            weight,
                            point_diff,
                            sum(win_susp),
                            sum(norm_diff),
                            testval,
                        )
                    else:
                        pass
                        print(
                            "suspicous points at ",
                            i + 1,
                            ": ",
                            weight,
                            point_diff,
                            sum(win_susp),
                            sum(norm_diff),
                            testval,
                        )
                else:
                    pre_susp_bool = False
                    win_susp.append(0)
                    win_susp.pop(0)
                    if outlier_flag[i + 1] == 1:
                        print(
                            "outlier weight at ",
                            i + 1,
                            ": NO ",
                            weight,
                            point_diff,
                            sum(win_susp),
                            sum(norm_diff),
                            testval,
                        )
                if i in ts_timestamp:
                    print(
                        "----------- CHANGE POINT AT -------",
                        i,
                        ":",
                        tmp_pro_conve,
                        weight,
                    )
                self.train_on_batch(tmp_cur_trnx, tmp_cur_trny * (weight))

            cur_pred = self.predict_on_batch(tmp_cur_trnx)[0][0][0]
            tmpresi = weight * (tmp_cur_trny[0][0][0] - cur_pred)
            resi_mean = resi_mean * seg_cnt * 1.0 / (seg_cnt + 1) + tmpresi / (
                seg_cnt + 1
            )
            resi_sqr = resi_sqr * seg_cnt * 1.0 / (
                seg_cnt + 1
            ) + tmpresi * tmpresi / (seg_cnt + 1)
            resi_var = resi_sqr - resi_mean * resi_mean
            seg_cnt = seg_cnt + 1

            # ----------------prediction-------------------------- #
            if sus_point_bool is True:
                vali_testx = copy.deepcopy(x[:, i : i + 1, :])
                vali_testx[0][0][0] = cur_pred_ini
            else:
                vali_testx = copy.deepcopy(x[:, i + 1 : i + 2, :])

            if outlier_flag[i + 2] != 1:
                pred_test = self.predict_on_batch(vali_testx)[0][0][0]
                y_orig.append(y[:, i + 1 : i + 2, :][0][0][0])
                y_pred.append(pred_test)

        print("Normalised MSE: ", nmse(y_orig, y_pred))

        # plot_series(
        #     y_orig,
        #     y_pred,
        #     plot_title="Online Prediction by WGSLTM on Bike rental",
        #     save_path="plot_wglstm.png",
        # )

        return SimpleNamespace(
            history={m.name: m.result() for m in self.metrics}
        )

    def sliding_window_features(self, cur_pos, dataX, dataY, susp_list):
        winsize = self.window_size
        resi_list = []
        for i in range(cur_pos - winsize, cur_pos):
            cur_trnx = dataX[:, i : i + 1, :]
            cur_trny = dataY[:, i : i + 1, :]
            cur_pred = self.predict_on_batch(cur_trnx)[0][0][0]
            tmpresi = cur_trny[0][0][0] - cur_pred * 1.0
            resi_list.append(tmpresi)

        # extract features
        tmparr = list(zip(range(winsize), susp_list))
        lnormal = -1
        rnormal = -1
        rsus = -1

        tmpdiff = []
        for i in range(cur_pos - winsize, cur_pos):
            cur_val = dataX[:, i : i + 1, :][0][0][0]
            if susp_list[i - (cur_pos - winsize)] == 0:
                if lnormal == -1:
                    lnormal = i
                    tmpdiff.append(0.0)
                else:
                    tmpdiff.append(
                        abs(
                            cur_val
                            - dataX[:, lnormal : lnormal + 1, :][0][0][0]
                        )
                    )
                    lnormal = i
            else:
                rsus = i
                if i >= (cur_pos - winsize):
                    tmpdiff.append(
                        abs(cur_val - dataX[:, i - 1 : i, :][0][0][0])
                    )
                else:
                    tmpdiff.append(abs(0.0))

        for i in range(cur_pos - winsize, cur_pos):
            tmp_dta = cur_pos - (i - (cur_pos - winsize)) - 1
            cur_val = dataX[:, tmp_dta : tmp_dta + 1, :][0][0][0]
            tmp_win = tmp_dta - (cur_pos - winsize)
            if susp_list[tmp_win] == 0:
                if rnormal == -1:
                    rnormal = tmp_dta
                else:
                    tmpdiff[tmp_win] = tmpdiff[tmp_win] + (
                        abs(
                            cur_val
                            - dataX[:, rnormal : rnormal + 1, :][0][0][0]
                        )
                    )
                    rnormal = tmp_dta
            else:
                if tmp_dta <= (cur_pos - 1):
                    tmpdiff[tmp_win] = tmpdiff[tmp_win] + (
                        abs(
                            cur_val
                            - dataX[:, tmp_dta + 1 : tmp_dta + 2, :][0][0][0]
                        )
                    )

        tmparr = list(zip(tmpdiff, susp_list))
        norm_diff = [i[0] for i in tmparr if i[1] == 0]
        susp_diff = [i[0] for i in tmparr if i[1] == 1]

        tmpdiff = -1
        if rsus != -1:
            tmpdiff = abs(
                dataX[:, cur_pos : cur_pos + 1, :][0][0][0]
                - dataX[:, rsus : rsus + 1, :][0][0][0]
            )
        tmpval = abs(
            dataX[:, cur_pos : cur_pos + 1, :][0][0][0]
            - dataX[:, cur_pos - 1 : cur_pos, :][0][0][0]
        )
        tmpdiff = max(tmpdiff, tmpval)

        return norm_diff, susp_diff, tmpdiff

    def sliding_window_weight(self, normDiff, suspDiff, pnt_diff):
        weight_mag = 5
        weight_beta = 0.05

        if len(suspDiff) != 0 and len(normDiff) != 0:
            r2 = (
                1.0
                * sum(suspDiff)
                * len(normDiff)
                / sum(normDiff)
                / len(suspDiff)
            )
            if r2 < 5:
                r2 = 0.0

        if len(normDiff) != 0:
            tmpr = pnt_diff * 1.0 / (sum(normDiff) / len(normDiff))
            if tmpr > weight_mag:
                tmpw = np.exp(-1 * weight_beta * tmpr)
            else:
                tmpw = 1.0
        else:
            tmpw = 1.0

        return tmpw


class WGLSTM(KerasRegressor, KerasSerializationMixin):
    """
    Keras wrapper for Weighted Gradient learning based LSTM (WGLSTM)
    neural network in online learning of time series.

    WGLSTM is an adaptive gradient learning method for recurrent neural
    networks (RNN) to forecast streaming time series in the presence of
    anomalies and change points.

    It leverage local features, which are extracted from a sliding window
    over time series, to dynamically weight the gradient, at each time instant
    for updating the current neural network.

    Parameters
    ----------

        optimizer : Union[str, tf.keras.optimizers.Optimizer,
        Type[tf.keras.optimizers.Optimizer]], default "sgd"
            This can be a string for Keras' built in optimizersan instance of
            tf.keras.optimizers.Optimizer or a class inheriting from
            tf.keras.optimizers.Optimizer. Only strings and classes support
            parameter routing.

        loss : Union[Union[str, tf.keras.losses.Loss, Callable], None],
        default="mse"
            The loss function to use for training. This can be a string for
            Keras' built in losses, an instance of tf.keras.losses.Loss or a
            class inheriting from tf.keras.losses.Loss .

        metrics : List[str], default=None
            List of metrics to evaluate and report at each epoch.

        hidden_layer_activation : str, default=linear
            What activation to use on the hidden layer - can use any
            tf.keras.activation function name.

        hidden_layer_neurons: int, default=450
            number of neurons to use for learning in the hidden layer.

        regularization_factor: float, default=0.0001
            The regularization factor to used during the training.

        timesteps: int, default=1
            The amount of time steps to run your recurrent neural network.

        num_of_features : int, default=1
            Number of feature variables in every time step.

        window_size: int, default=20
            The size of the sliding window for feature extraction.

        epochs: int, default=1
            Number of training steps.

        verbose: int default=0
            0 means no output printed during training.

    """

    def __init__(
        self,
        loss="mse",
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=0.002, momentum=0.03, decay=0.0, nesterov=True
        ),
        metrics=None,
        epochs=1,
        verbose=0,
        num_of_features=1,
        hidden_layer_neurons=450,
        hidden_layer_activation="linear",
        regularization_factor=0.0001,
        timesteps=1,
        window_size=20,
        **kwargs,
    ) -> None:
        super(WGLSTM, self).__init__(
            _Model(
                num_of_features,
                hidden_layer_neurons,
                hidden_layer_activation,
                regularization_factor,
                timesteps,
                window_size,
            ),
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            epochs=epochs,
            verbose=verbose,
            **kwargs,
        )
