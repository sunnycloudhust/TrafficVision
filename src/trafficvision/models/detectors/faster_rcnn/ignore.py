from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from ....data.ignore import centers_in_regions


class IgnoreTransform(GeneralizedRCNNTransform):
    def resize(self, image, target=None):
        old_h, old_w = image.shape[-2:]
        image, target = super().resize(image, target)
        if target is not None and "ignored_regions" in target:
            new_h, new_w = image.shape[-2:]
            target["ignored_regions"] = target["ignored_regions"] * image.new_tensor(
                [new_w / old_w, new_h / old_h] * 2
            )
        return image, target


class IgnoreRPN(RegionProposalNetwork):
    def assign_targets_to_anchors(self, anchors, targets):
        labels, matched = super().assign_targets_to_anchors(anchors, targets)
        for boxes, target, label in zip(anchors, targets, labels):
            regions = target.get("ignored_regions", boxes.new_empty((0, 4)))
            label[centers_in_regions(boxes, regions) & (label != 1)] = -1
        return labels, matched


class IgnoreRoIHeads(RoIHeads):
    def select_training_samples(self, proposals, targets):
        # RoIHeads does not pass ignored regions to assign_targets_to_proposals.
        # Drop ignored external/RPN proposals before GT boxes are appended.
        proposals = [
            boxes[~centers_in_regions(boxes, target.get("ignored_regions", boxes.new_empty((0, 4))))]
            for boxes, target in zip(proposals, targets)
        ]
        return super().select_training_samples(proposals, targets)
