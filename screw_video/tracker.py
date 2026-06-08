"""
ByteTrack-based multi-object tracker for screws
"""
from boxmot import ByteTrack
import numpy as np

from reid import OSNetReID, ReIDTrackMatcher


class ScrewTracker:
    """Wrapper for ByteTrack multi-object tracker"""

    def __init__(
        self,
        track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=30,
        reid_weights_path=None,
        reid_model_name="osnet_x0_25",
        reid_match_thresh=0.75,
        reid_max_age=90,
        reid_update_interval=5,
        reid_device=None,
        edge_margin=0,
    ):
        """
        Args:
            track_thresh: Detection confidence threshold for tracking
            track_buffer: Number of frames to keep lost tracks
            match_thresh: Matching threshold for data association
            frame_rate: Video frame rate
        """
        self.tracker = ByteTrack(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate
        )
        self.edge_margin = edge_margin
        self.reid_matcher = None
        if reid_weights_path is not None:
            reid_extractor = OSNetReID(
                weights_path=reid_weights_path,
                model_name=reid_model_name,
                device=reid_device,
            )
            self.reid_matcher = ReIDTrackMatcher(
                feature_extractor=reid_extractor,
                match_thresh=reid_match_thresh,
                max_age=reid_max_age,
                update_interval=reid_update_interval,
            )
        # Track class history for voting
        self.track_class_history = {}  # {track_id: [class_id, class_id, ...]}

    def _filter_edge_detections(self, detections, frame_shape):
        if self.edge_margin <= 0:
            return detections

        frame_h, frame_w = frame_shape[:2]
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det['box']
            # Border-touching boxes are often truncated objects and tend to create unstable tracks.
            if x1 <= self.edge_margin:
                continue
            if y1 <= self.edge_margin:
                continue
            if x2 >= frame_w - self.edge_margin:
                continue
            if y2 >= frame_h - self.edge_margin:
                continue
            filtered.append(det)

        return filtered

    def update(self, detections, frame):
        """
        Update tracker with new detections

        Args:
            detections: List of {'box': [x1,y1,x2,y2], 'class': int, 'score': float}

        Returns:
            List of tracks: [{'box': [x1,y1,x2,y2], 'class': int, 'track_id': int}, ...]
        """
        detections = self._filter_edge_detections(detections, frame.shape)

        if not detections:
            empty = np.zeros((0, 6), dtype=np.float32)
            self.tracker.update(empty, frame)
            return []

        dets = np.array([
            [d['box'][0], d['box'][1], d['box'][2], d['box'][3], d['score'], d['class']]
            for d in detections
        ], dtype=np.float32)

        tracks = self.tracker.update(dets, frame)

        results = []
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls, det_ind = track
            track_id = int(track_id)
            class_id = int(cls)

            results.append({
                'box': [x1, y1, x2, y2],
                'class': class_id,
                'track_id': track_id
            })

        if self.reid_matcher is not None and results:
            results = self.reid_matcher.update(frame, results)

        for track in results:
            track_id = track['track_id']
            class_id = track['class']
            if track_id not in self.track_class_history:
                self.track_class_history[track_id] = []
            self.track_class_history[track_id].append(class_id)

        return results

    def get_track_final_class(self, track_id):
        """
        Get the final class for a track using majority voting

        Args:
            track_id: Track ID

        Returns:
            Most common class_id in the track's history, or None if not found
        """
        if track_id not in self.track_class_history:
            return None

        from collections import Counter
        class_history = self.track_class_history[track_id]
        if not class_history:
            return None

        # Return the most common class
        return Counter(class_history).most_common(1)[0][0]

    def get_track_length(self, track_id):
        """Get the number of frames associated with a track."""
        if track_id not in self.track_class_history:
            return 0
        return len(self.track_class_history[track_id])
