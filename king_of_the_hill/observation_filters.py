from ray.rllib.utils import Filter


class RGBFilter(Filter):
    """
    Observation filter that normalizes rgb observations to range from 0 to 1
    """

    def __init__(self, *args, **kwargs):
        self.rgb_max_val = 255

    def reset_buffer(self) -> None:
        pass

    def apply_changes(self, other, *args, **kwargs):
        pass

    def copy(self):
        return RGBFilter()

    def as_serializable(self):
        return RGBFilter()

    def sync(self, other):
        pass

    def __call__(self, x, update=True):
        return x/self.rgb_max_val

    def __repr__(self):
        return f"RGBFilter()"
