from torch.utils.tensorboard import SummaryWriter


class TensorboardWriter(SummaryWriter):
    def __init__(self, log_dir=None, *args, **kwrags):
        super().__init__(log_dir, *args, **kwrags)
        self.index = -1

    def write_predictions(self, y_pred, y_true):
        for yh, y_true in zip(y_pred, y_true):
            self.add_scalars(
                "Predictions",
                {
                    "y_true": yh,
                    "y_pred": y_true,
                },
                self.index,
            )
            self.index += 1
            # self.flush()

    def write_score(self, score, epoch_n):
        self.add_scalar(
            tag="Score", scalar_value=score, global_step=epoch_n, new_style=True
        )

        if epoch_n % 10 == 0:
            self.add_scalar(
                tag="Score", scalar_value=score, global_step=epoch_n, new_style=True
            )
        # self.flush()
