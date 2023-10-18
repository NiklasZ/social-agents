from ray.tune import Callback


class PrintResultCallback(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        print(f"Trial {trial} in iteration {iteration}, "
              f"completed training steps: {result['training_iteration']}")
