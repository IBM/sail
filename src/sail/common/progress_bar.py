from tqdm import tqdm

BAR_FORMAT = {
    "pipeline_training": "{desc}: {percentage:3.0f}%{bar} [Steps: {n_fmt}/{total_fmt}, ETA: {elapsed}<{remaining}, Elapsed:{elapsed_s:3.3f}s{postfix}]",
    "model_training": "{desc}: {percentage:3.0f}%{bar} [ETA: {elapsed}<{remaining}, Elapsed:{elapsed_s:3.3f}s{postfix}]",
    "scoring": "{desc}: {percentage:3.0f}%{bar} [Points: {n_fmt}/{total_fmt}, Elapsed:{elapsed_s:3.4f}s{postfix}]",
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
