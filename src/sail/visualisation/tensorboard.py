import torch
from torch.utils.tensorboard import SummaryWriter, FileWriter
import time
from torch.utils.tensorboard.summary import scalar
import numpy as np
import glob
import os


class TensorboardWriter(SummaryWriter):
    def __init__(
        self, log_dir=None, exp_name="SAIL_Training", use_dir=None, *args, **kwrags
    ):
        if use_dir:
            exp_dir = log_dir + "/" + use_dir
        else:
            dirs = sorted(
                glob.glob(os.path.join(log_dir, exp_name + "_*")), reverse=True
            )
            new_dir_no = 1
            if len(dirs) > 0:
                new_dir_no = int(dirs[0].split("_")[-1]) + 1
            exp_dir = log_dir + "/" + exp_name + "_" + str(new_dir_no)
        super().__init__(exp_dir, *args, **kwrags)
        self.all_writers = {}

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
        self.add_scalars_custom(
            "Score_and_Detect",
            {"Score": score, "Drift": score},
            epoch_n,
            include_main_tag=False,
        )

        if drift_point:
            self.add_scalars_custom(
                "Score_and_Detect",
                {
                    "Drift": np.nan,
                },
                epoch_n,
                include_main_tag=False,
            )
            self.add_scalars_custom(
                "Score_and_Detect",
                {
                    "Drift": score,
                },
                epoch_n,
                include_main_tag=False,
            )
        self.flush()

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
