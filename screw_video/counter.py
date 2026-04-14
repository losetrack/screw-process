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
        self.track_history = {}  # {track_id: {class_id: votes}}

    def update(self, tracks):
        """
        Update counter with new tracks

        Args:
            tracks: List of {'box': [...], 'class': int, 'track_id': int}
        """
        for track in tracks:
            tid = track['track_id']
            class_id = track['class']

            if tid not in self.track_history:
                self.track_history[tid] = {}

            self.track_history[tid][class_id] = self.track_history[tid].get(class_id, 0) + 1

    def get_counts(self):
        """
        Get final counts for all classes

        Returns:
            List of counts [Type_1, Type_2, Type_3, Type_4, Type_5]
        """
        counts = [0] * self.num_classes
        for class_votes in self.track_history.values():
            class_id = max(class_votes, key=class_votes.get)
            if 0 <= class_id < self.num_classes:
                counts[class_id] += 1
        return counts

    def reset(self):
        """Reset counter for new video"""
        self.track_history.clear()
