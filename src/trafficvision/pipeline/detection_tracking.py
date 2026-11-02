from ..geometry.coordinates import bottom_center
from ..speed.estimator import SpeedEstimator
from ..zones.zone_manager import ZoneManager


class DetectionTrackingPipeline:
    def __init__(self, detector, tracker, zones=None, homography=None, speed_config=None):
        self.detector, self.tracker = detector, tracker
        self.zones = zones or ZoneManager()
        self.homography = homography
        self.speed = SpeedEstimator(**(speed_config or {}))

    def process(self, image, timestamp, frame=None):
        detections = self.zones.detect(self.detector, image)
        tracks = self.tracker.update(detections, frame)
        self.speed.prune(timestamp)
        results = []
        for track in tracks:
            assigned = self.zones.assignments(track.box)
            world, kmh = None, None
            measure = any(z.measure_speed for z in assigned) if self.zones.zones else True
            if self.homography is not None and measure:
                try:
                    world = self.homography.project(bottom_center(track.box))
                    kmh = self.speed.update(track.track_id, world, timestamp)
                except ValueError:  # A horizon point is not a world observation.
                    self.speed.reset(track.track_id)
            else:
                self.speed.reset(track.track_id)
            results.append(dict(track=track, world=world, speed_kmh=kmh, zones=[z.name for z in assigned]))
        return results
