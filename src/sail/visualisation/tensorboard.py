import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.tensorboard import FileWriter, SummaryWriter
from torch.utils.tensorboard._utils import figure_to_image
from torch.utils.tensorboard.summary import image, scalar

from sail.utils.logging import configure_logger

LOGGER = configure_logger(logger_name="TensorboardWriter")


class TensorboardWriter(SummaryWriter):
    def __init__(
        self, log_dir, exp_name="Training_Logs", use_dir=None, *args, **kwrags
    ):
        if use_dir:
            exp_dir = log_dir + "/" + use_dir
        else:
            dirs = sorted(
                glob.glob(os.path.join(log_dir, exp_name, "v*")), reverse=True
            )
            new_dir_no = 1
            if len(dirs) > 0:
                new_dir_no = int(dirs[0].split("/")[-1][1]) + 1
            exp_dir = os.path.join(log_dir, exp_name, "v" + str(new_dir_no))
            # log_dir + "/" + exp_name + "/_v" + str(new_dir_no)

        super().__init__(exp_dir, *args, **kwrags)
        self.all_writers = {}

        LOGGER.info(
            f"Sending training output to Tensorboard logs. Please run `tensorboard --logdir {os.path.join(log_dir, exp_name)}` in terminal to start tensorboard server and track training progress."
        )

    def get_state(self):
        return {"log_dir": self.log_dir}

    def set_state(self, state):
        self.log_dir = state["log_dir"]

    def _get_file_writer(self):
        ...

    def write_predictions(self, y_pred, y_true, start_index):
        for index, (yh, y_true) in enumerate(zip(y_pred, y_true), start=start_index):
            self.add_scalars_custom(
                "Predictions",
                {
                    "y_true": yh,
                    "y_pred": y_true,
                },
                index,
            )

    def write_score(self, score, epoch_n, drift_point=False):
        self.add_scalar(
            "Score_and_Detect",
            score,
            epoch_n,
        )

        if drift_point:
            self.add_scalar(
                "Score_and_Detect",
                np.nan,
                epoch_n,
            )
            self.add_scalar(
                "Score_and_Detect",
                score,
                epoch_n,
            )
        self.flush()

    def write_classification_report(self, y_pred, y_true, epoch_n):
        figure, (ax1, ax2) = plt.subplots(2, 1)
        figure.set_size_inches(6, 6)
        figure.set_dpi(150)

        labels = np.unique(y_true).tolist()
        cf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
        sns.heatmap(
            cf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax1,
            xticklabels=labels,
            yticklabels=labels,
        )
        ax1.title.set_text("Confusion Matrix")

        sns.heatmap(
            pd.DataFrame(
                classification_report(y_true, y_pred, labels=labels, output_dict=True)
            )
            .iloc[:-1, :]
            .T,
            annot=True,
            ax=ax2,
        )
        ax2.title.set_text("Classfication Report")

        self.add_image_custom(
            tag="Confusion_matrix",
            figure=figure,
            global_step=epoch_n,
        )
        plt.close()
        self.flush()

    def add_image_custom(
        self,
        tag,
        figure,
        global_step=None,
        walltime=None,
        close=True,
        dataformats="CHW",
    ):
        fw_tag = self.log_dir + "/" + tag
        if self.all_writers and fw_tag in self.all_writers.keys():
            fw = self.all_writers[fw_tag]
        else:
            fw = FileWriter(
                fw_tag, self.max_queue, self.flush_secs, self.filename_suffix
            )
            self.all_writers[fw_tag] = fw

        fw.add_summary(
            image(tag, figure_to_image(figure, close), dataformats=dataformats),
            global_step,
            walltime,
        )

    def add_scalar(
        self,
        tag,
        scalar_value,
        global_step=None,
        walltime=None,
        new_style=False,
        double_precision=False,
    ):
        """Add scalar data to summary.

        Args:
            tag (str): Data identifier
            scalar_value (float or string/blobname): Value to save
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
            with seconds after epoch of event
            new_style (boolean): Whether to use new style (tensor field) or old
            style (simple_value field). New style could lead to faster data loading.
        Examples::

            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            x = range(100)
            for i in x:
                writer.add_scalar('y=2x', i * 2, i)
            writer.close()

        Expected result:

        .. image:: _static/img/tensorboard/add_scalar.png
        :scale: 50 %

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_scalar")
        summary = scalar(
            tag, scalar_value, new_style=new_style, double_precision=double_precision
        )
        fw_tag = self.log_dir + "/" + tag
        if self.all_writers and fw_tag in self.all_writers.keys():
            fw = self.all_writers[fw_tag]
        else:
            fw = FileWriter(
                fw_tag, self.max_queue, self.flush_secs, self.filename_suffix
            )
            self.all_writers[fw_tag] = fw

        fw.add_summary(summary, global_step, walltime)

    def add_scalars_custom(
        self,
        main_tag,
        tag_scalar_dict,
        global_step=None,
        walltime=None,
        include_main_tag=True,
    ):
        """Adds many scalar data to summary.

        Args:
            main_tag (str): The parent name for the tags
            tag_scalar_dict (dict): Key-value pair storing the tag and corresponding values
            global_step (int): Global step value to record
            walltime (float): Optional override default walltime (time.time())
              seconds after epoch of event

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter()
            r = 5
            for i in range(100):
                writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
                                                'xcosx':i*np.cos(i/r),
                                                'tanx': np.tan(i/r)}, i)
            writer.close()
            # This call adds three values to the same scalar plot with the tag
            # 'run_14h' in TensorBoard's scalar section.

        Expected result:

        .. image:: _static/img/tensorboard/add_scalars.png
           :scale: 50 %

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_scalars")
        walltime = time.time() if walltime is None else walltime
        for tag, scalar_value in tag_scalar_dict.items():
            if include_main_tag:
                fw_tag = self.log_dir + "/" + main_tag + "_" + tag
            else:
                fw_tag = self.log_dir + "/" + tag
            if self.all_writers and fw_tag in self.all_writers.keys():
                fw = self.all_writers[fw_tag]
            else:
                fw = FileWriter(
                    fw_tag, self.max_queue, self.flush_secs, self.filename_suffix
                )
                self.all_writers[fw_tag] = fw

            fw.add_summary(scalar(main_tag, scalar_value), global_step, walltime)
