"""
Screw counter based on track IDs
"""


class ScrewCounter:
    """Count screws by tracking unique track IDs"""

    def __init__(self, num_classes=5):
        """
        Args:
            num_classes: Number of screw classes (default: 5)
        """
        self.num_classes = num_classes
        self.track_history = {}  # {track_id: class_id}

    def update(self, tracks):
        """
        Update counter with new tracks

        Args:
            tracks: List of {'box': [...], 'class': int, 'track_id': int}
        """
        for track in tracks:
            tid = track['track_id']
            if tid not in self.track_history:
                # Record the class of this track when first seen
                self.track_history[tid] = track['class']

    def get_counts(self):
        """
        Get final counts for all classes

        Returns:
            List of counts [Type_1, Type_2, Type_3, Type_4, Type_5]
        """
        counts = [0] * self.num_classes
        for class_id in self.track_history.values():
            if 0 <= class_id < self.num_classes:
                counts[class_id] += 1
        return counts

    def reset(self):
        """Reset counter for new video"""
        self.track_history.clear()
