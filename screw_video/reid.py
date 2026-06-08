"""
OSNet-based Re-ID utilities for screw tracking.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch


class OSNetReID:
    """Extract appearance embeddings with an OSNet backbone."""

    def __init__(
        self,
        weights_path,
        model_name="osnet_x0_25",
        device=None,
        image_size=(128, 256),
    ):
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Re-ID weights not found: {self.weights_path}")

        try:
            import torchreid
        except ImportError as exc:
            raise ImportError("torchreid is required for OSNet Re-ID") from exc

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = tuple(image_size)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1,
            pretrained=False,
            use_gpu=self.device.type == "cuda",
        )
        torchreid.utils.load_pretrained_weights(self.model, str(self.weights_path))
        self.model.to(self.device)
        self.model.eval()

    def extract(self, frame, boxes):
        if not boxes:
            return np.empty((0, 0), dtype=np.float32), []

        crops = []
        valid_indices = []
        frame_h, frame_w = frame.shape[:2]

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            x1 = int(np.clip(np.floor(x1), 0, frame_w - 1))
            y1 = int(np.clip(np.floor(y1), 0, frame_h - 1))
            x2 = int(np.clip(np.ceil(x2), 0, frame_w))
            y2 = int(np.clip(np.ceil(y2), 0, frame_h))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, self.image_size, interpolation=cv2.INTER_LINEAR)
            crop = crop.astype(np.float32) / 255.0
            crop = (crop - self.mean) / self.std
            crop = np.transpose(crop, (2, 0, 1))
            crops.append(crop)
            valid_indices.append(idx)

        if not crops:
            return np.empty((0, 0), dtype=np.float32), []

        batch = torch.from_numpy(np.stack(crops)).to(self.device)
        with torch.no_grad():
            features = self.model(batch)
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        return features.cpu().numpy().astype(np.float32), valid_indices


class ReIDTrackMatcher:
    """Maintain global IDs by matching OSNet embeddings over time."""

    def __init__(
        self,
        feature_extractor,
        match_thresh=0.75,
        max_age=90,
        feature_history=10,
        update_interval=5,
    ):
        self.feature_extractor = feature_extractor
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.feature_history = feature_history
        self.update_interval = max(1, int(update_interval))

        self.frame_index = 0
        self.next_global_id = 1
        self.local_to_global = {}
        self.global_tracks = {}
        self.active_local_ids = set()

    def update(self, frame, tracks):
        current_local_ids = {track["track_id"] for track in tracks}
        disappeared_local_ids = self.active_local_ids - current_local_ids
        for local_id in disappeared_local_ids:
            self.local_to_global.pop(local_id, None)

        self.active_local_ids = current_local_ids
        used_global_ids = set()

        request_indices = []
        request_boxes = []
        new_track_indices = []

        for idx, track in enumerate(tracks):
            local_id = track["track_id"]
            if local_id not in self.local_to_global:
                # New local tracks always need an appearance feature for global ID assignment.
                new_track_indices.append(idx)
                request_indices.append(idx)
                request_boxes.append(track["box"])
                continue

            global_id = self.local_to_global[local_id]
            track_state = self.global_tracks.get(global_id)
            if track_state is None:
                # Recover gracefully if the local-global mapping exists but the global state was dropped.
                new_track_indices.append(idx)
                request_indices.append(idx)
                request_boxes.append(track["box"])
                continue

            last_feature_frame = track_state.get("last_feature_frame", -self.update_interval)
            if self.frame_index - last_feature_frame >= self.update_interval:
                # Existing tracks refresh features sparsely to trade a small amount of accuracy for speed.
                request_indices.append(idx)
                request_boxes.append(track["box"])

        features, valid_indices = self.feature_extractor.extract(frame, request_boxes)
        feature_map = {
            request_indices[track_idx]: features[i]
            for i, track_idx in enumerate(valid_indices)
        }
        for idx, track in enumerate(tracks):
            local_id = track["track_id"]
            class_id = track["class"]
            feature = feature_map.get(idx)
            if local_id in self.local_to_global:
                global_id = self.local_to_global[local_id]
                track_state = self.global_tracks.get(global_id)
                if track_state is not None:
                    if feature is not None:
                        self._update_global_track(global_id, feature, class_id)
                    else:
                        self._touch_global_track(global_id)
                track["track_id"] = global_id
                used_global_ids.add(global_id)

        for idx in new_track_indices:
            track = tracks[idx]
            local_id = track["track_id"]
            class_id = track["class"]
            feature = feature_map.get(idx)

            # New ByteTrack IDs are matched against historical global IDs here.
            global_id = self._match_existing_track(feature, class_id, used_global_ids)

            if global_id is None:
                # If no historical track matches, start a new global identity.
                global_id = self._create_global_track(feature, class_id)
            else:
                # A successful match also refreshes that identity's appearance prototype.
                self._update_global_track(global_id, feature, class_id)

            self.local_to_global[local_id] = global_id
            track["track_id"] = global_id
            used_global_ids.add(global_id)

        self.frame_index += 1
        return tracks

    def _match_existing_track(self, feature, class_id, used_global_ids):
        if feature is None:
            return None

        best_global_id = None
        best_similarity = self.match_thresh

        for global_id, track_state in self.global_tracks.items():
            if global_id in used_global_ids:
                continue
            if self.frame_index - track_state["last_seen"] > self.max_age:
                continue
            if track_state["class_id"] != class_id:
                continue

            # Matching is restricted to same-class historical identities with a valid appearance prototype.
            if track_state["prototype"] is None:
                continue

            similarity = float(np.dot(feature, track_state["prototype"]))
            if similarity > best_similarity:
                best_similarity = similarity
                best_global_id = global_id

        return best_global_id

    def _create_global_track(self, feature, class_id):
        global_id = self.next_global_id
        self.next_global_id += 1
        self.global_tracks[global_id] = {
            "class_id": class_id,
            "prototype": feature,
            "features": deque([feature] if feature is not None else [], maxlen=self.feature_history),
            "last_seen": self.frame_index,
            "last_feature_frame": self.frame_index if feature is not None else -self.update_interval,
        }
        return global_id

    def _update_global_track(self, global_id, feature, class_id):
        track_state = self.global_tracks[global_id]
        track_state["last_seen"] = self.frame_index

        if feature is None:
            return

        track_state["features"].append(feature)
        stacked = np.stack(track_state["features"])
        # Keep a short moving prototype so appearance can adapt without drifting too fast.
        prototype = stacked.mean(axis=0)
        norm = np.linalg.norm(prototype)
        if norm > 0:
            prototype = prototype / norm
        track_state["prototype"] = prototype.astype(np.float32)
        track_state["last_feature_frame"] = self.frame_index

    def _touch_global_track(self, global_id):
        self.global_tracks[global_id]["last_seen"] = self.frame_index
