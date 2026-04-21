"""
Screw counter based on track IDs with majority voting
"""
class ScrewCounter:
    """Count screws by tracking unique track IDs"""

    def __init__(self, num_classes=5, min_track_length=1):
        """
        Args:
            num_classes: Number of screw classes (default: 5)
        """
        self.num_classes = num_classes
        self.min_track_length = min_track_length
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
        Get final counts for all classes using majority voting

        Returns:
            List of counts [Type_1, Type_2, Type_3, Type_4, Type_5]
        """
        counts = [0] * self.num_classes
        for class_votes in self.track_history.values():
            class_id = max(class_votes, key=class_votes.get)
            if 0 <= class_id < self.num_classes:
                counts[class_id] += 1
        return counts

    def get_counts_with_voting(self, tracker):
        """
        Get final counts using tracker's class history

        Args:
            tracker: ScrewTracker instance with track_class_history

        Returns:
            List of counts [Type_1, Type_2, Type_3, Type_4, Type_5]
        """
        counts = [0] * self.num_classes

        # Use tracker's class history for more accurate voting
        for track_id in self.track_history.keys():
            if tracker.get_track_length(track_id) < self.min_track_length:
                continue
            final_class = tracker.get_track_final_class(track_id)
            if final_class is not None and 0 <= final_class < self.num_classes:
                counts[final_class] += 1

        return counts

    def reset(self):
        """Reset counter for new video"""
        self.track_history.clear()
