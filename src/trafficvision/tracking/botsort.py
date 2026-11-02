from .bytetrack import UltralyticsTracker


class BoTSORT(UltralyticsTracker):
    def __init__(self, config=None, fps=25):
        super().__init__("botsort", config, fps)
