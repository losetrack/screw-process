"""
Visualization utilities for screw tracking
"""
import cv2
import numpy as np


# Color palette for 5 screw classes (BGR format)
COLORS = [
    (255, 0, 0),      # Type_1: Blue
    (0, 255, 0),      # Type_2: Green
    (0, 0, 255),      # Type_3: Red
    (255, 255, 0),    # Type_4: Cyan
    (255, 0, 255)     # Type_5: Magenta
]


def draw_tracks(frame, tracks, alpha=0.4):
    """
    Draw tracking results on frame with colored bounding boxes

    Args:
        frame: Input image (BGR format)
        tracks: List of {'box': [x1,y1,x2,y2], 'class': int, 'track_id': int}
        alpha: Transparency for overlay (0=transparent, 1=opaque)

    Returns:
        Annotated image
    """
    overlay = frame.copy()

    for track in tracks:
        x1, y1, x2, y2 = map(int, track['box'])
        class_id = track['class']
        track_id = track['track_id']

        # Get color for this class
        color = COLORS[class_id % len(COLORS)]

        # Draw bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Draw label with track ID and class
        label = f"ID:{track_id} Type:{class_id + 1}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_w, label_h = label_size

        # Draw label background
        cv2.rectangle(overlay, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

        # Draw label text
        cv2.putText(overlay, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Blend overlay with original frame
    result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

    return result
