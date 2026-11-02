from ..faster_rcnn.model import FasterRCNN
from .proposals import ExternalProposals


class FastRCNN(FasterRCNN):
    """Shared image CNN + external proposals + RoIAlign + trainable class/box heads."""

    def __init__(self, proposal_strategy="selective_search", proposals=300, **kwargs):
        super().__init__(proposals=proposals, **kwargs)
        self.rpn = ExternalProposals(proposal_strategy, proposals)
