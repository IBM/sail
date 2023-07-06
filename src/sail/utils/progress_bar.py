import os
from threading import Thread
from time import sleep
from typing import Any

import ray
from tqdm import tqdm

BAR_FORMAT = {
    "pipeline_training": "{desc}: {percentage:3.0f}%{bar} [Steps: {n_fmt}/{total_fmt}, ETA: {elapsed}<{remaining}, Elapsed:{elapsed_s:3.3f}s{postfix}]",
    "model_training": "{desc}: {percentage:3.0f}%{bar} [ETA: {elapsed}<{remaining}, Elapsed:{elapsed_s:3.3f}s{postfix}]",
    "scoring": "{desc}: {percentage:3.0f}%{bar} [Points: {n_fmt}/{total_fmt}, Elapsed:{elapsed_s:3.4f}s{postfix}]",
    "tuning": "{desc} [Elapsed: {elapsed_s:3.2f}s{postfix}]",
    "progressive_score": "{desc} [Elapsed: {elapsed_s:3.5f}s{postfix}]",
}


class SAILProgressBar:
    def __init__(self, steps, desc, params, format, verbose=1) -> None:
        self.remaining_steps = steps
        self.desc = desc
        self.params = params
        self.verbose = verbose

        if self.check_verbosity():
            self.pbar = tqdm(
                total=steps,
                ascii=" =",
                bar_format=BAR_FORMAT[format],
            )
            self.pbar.set_description_str(desc, refresh=True)
            self.pbar.set_postfix(params, refresh=True)

    def check_verbosity(self):
        return self.verbose == 1

    def append_desc(self, desc, divider=" "):
        if self.check_verbosity():
            self.pbar.set_description_str(self.desc + divider + desc, refresh=True)

    def update(self, value=1):
        if self.check_verbosity():
            self.remaining_steps -= value
            self.pbar.update(value)

    def update_params(self, key, value):
        if self.check_verbosity():
            self.params[key] = value
            self.pbar.set_postfix(self.params, refresh=True)

    def finish(self):
        if self.check_verbosity():
            self.pbar.update(self.remaining_steps)

    def refresh(self):
        self.pbar.refresh()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.check_verbosity():
            self.append_desc(desc="", divider="")
            return self.pbar.close()

    def close(self):
        if self.check_verbosity():
            self.pbar.close()


class SailTuningProgressBar(Thread):
    def __init__(self, search_method, warm_start) -> None:
        Thread.__init__(self)
        self.search_method = search_method
        self.num_trails = (
            search_method._list_grid_num_samples(warm_start)
            if isinstance(search_method.param_grid, list)
            else 0
        )
        self.exp_dir = os.path.expanduser(
            os.path.join(search_method.local_dir, search_method.name)
        )
        self.is_running = False

    def run(self):
        self.is_running = True
        params = {
            "Trials": "Running",
            "Nodes": len(ray.nodes()),
            "Cluster CPU": ray.cluster_resources()["CPU"],
            "Cluster Memory": str(
                format(ray.cluster_resources()["memory"] / (1024 * 1024 * 1024), ".2f")
            )
            + " GB",
        }

        progress = SAILProgressBar(
            steps=1,
            desc=f"SAIL Pipeline Tuning in progress...",
            params=params,
            format="tuning",
            verbose=1,
        )

        while True:
            progress.update()
            trial_terminated = []
            do_not_update = False
            try:
                for dir in os.listdir(self.exp_dir):
                    if dir not in trial_terminated:
                        file_path = self.exp_dir + "/" + dir + "/result.json"
                        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                            trial_terminated.append(dir)
            except Exception as e:
                do_not_update = True

            if not do_not_update:
                if self.num_trails > 0:
                    progress.update_params(
                        "Trials", f"{len(trial_terminated)}/{self.num_trails}"
                    )
                else:
                    progress.update_params("Trials", f"{len(trial_terminated)}")

            if not self.is_running:
                if do_not_update:
                    progress.update_params("Trials", "TERMINATED")

                progress.finish()
                progress.close()
                break

    def stop(self):
        self.is_running = False
