"""
ByteTrack-based multi-object tracker for screws
"""
from boxmot import BYTETracker
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
        self.tracker = BYTETracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate
        )

    def update(self, detections):
        """
        Update tracker with new detections

        Args:
            detections: List of {'box': [x1,y1,x2,y2], 'class': int, 'score': float}

        Returns:
            List of tracks: [{'box': [x1,y1,x2,y2], 'class': int, 'track_id': int}, ...]
        """
        if not detections:
            # Update with empty detections to handle lost tracks
            self.tracker.update(np.empty((0, 6)), None)
            return []

        # Convert to [x1,y1,x2,y2,score,class] format
        dets = np.array([
            [d['box'][0], d['box'][1], d['box'][2], d['box'][3], d['score'], d['class']]
            for d in detections
        ])

        # Update tracker: returns [x1,y1,x2,y2,track_id,class,...]
        tracks = self.tracker.update(dets, None)

        # Convert to output format
        results = []
        if len(tracks) > 0:
            for track in tracks:
                results.append({
                    'box': track[:4],
                    'class': int(track[5]),
                    'track_id': int(track[4])
                })

        return results
