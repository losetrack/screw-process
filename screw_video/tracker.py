"""
ByteTrack-based multi-object tracker for screws
"""
from boxmot import ByteTrack
import numpy as np


class ScrewTracker:
    """Wrapper for ByteTrack multi-object tracker"""

    def __init__(self, track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30):
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
        # Track class history for voting
        self.track_class_history = {}  # {track_id: [class_id, class_id, ...]}

    def update(self, detections, frame):
        """
        Update tracker with new detections

        Args:
            detections: List of {'box': [x1,y1,x2,y2], 'class': int, 'score': float}

        Returns:
            List of tracks: [{'box': [x1,y1,x2,y2], 'class': int, 'track_id': int}, ...]
        """
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

            # Record class history for this track
            if track_id not in self.track_class_history:
                self.track_class_history[track_id] = []
            self.track_class_history[track_id].append(class_id)

            results.append({
                'box': [x1, y1, x2, y2],
                'class': class_id,
                'track_id': track_id
            })

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
