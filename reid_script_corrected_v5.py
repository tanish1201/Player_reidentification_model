import cv2
from ultralytics import YOLO
import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment

# --- Kalman Filter for Object Tracking ---
class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2) # 4 states (x, y, vx, vy), 2 measurements (x, y)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        self.kf.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1

    def predict(self, x, y):
        # Update state with current measurement
        # Convert to float32 explicitly to avoid DeprecationWarning
        self.kf.correct(np.array([[np.float32(x)], [np.float32(y)]]))
        # Predict next state
        predicted = self.kf.predict()
        # Access elements explicitly to avoid DeprecationWarning
        return int(predicted[0][0]), int(predicted[1][0])

    def setup(self, bbox):
        # Initialize state with bounding box center
        x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        center_x, center_y = x + w/2, y + h/2
        self.kf.statePost = np.array([[center_x], [center_y], [0], [0]], np.float32)

# --- Tracking and Re-identification Logic ---
class PlayerTracker:
    def __init__(self, max_lost_frames=60, iou_threshold=0.3, cost_threshold=0.7):
        self.active_players = {}
        self.lost_players = {}
        self.next_player_id = 0
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        self.cost_threshold = cost_threshold # Max cost for assignment

    def _bbox_to_center(self, bbox):
        return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

    def _iou(self, box1, box2):
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)
        inter_area = inter_width * inter_height

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def _extract_features(self, frame_img, bbox):
        # A very simple feature: average color of the bounding box region
        x1, y1, x2, y2 = [int(c) for c in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_img.shape[1], x2), min(frame_img.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(3) # Return black if invalid bbox

        roi = frame_img[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros(3)
        return np.mean(roi, axis=(0, 1))

    def _compare_features(self, features1, features2):
        # Simple Euclidean distance for color features (lower is better)
        distance = np.linalg.norm(features1 - features2)
        # Convert distance to a similarity score (higher is better)
        # Max distance for 255 color channels is sqrt(3 * 255^2) approx 441
        max_dist = np.sqrt(3 * 255**2)
        return 1 - (distance / max_dist)

    def update(self, detections, frame_idx, frame_img):
        # 1. Predict next positions for active players
        predicted_tracks = []
        active_player_ids = list(self.active_players.keys())
        for player_id in active_player_ids:
            player_data = self.active_players[player_id]
            center_x, center_y = self._bbox_to_center(player_data["bbox"])
            predicted_center_x, predicted_center_y = player_data["kf"].predict(center_x, center_y)
            
            width = player_data["bbox"][2] - player_data["bbox"][0]
            height = player_data["bbox"][3] - player_data["bbox"][1]
            predicted_bbox = [
                predicted_center_x - width / 2,
                predicted_center_y - height / 2,
                predicted_center_x + width / 2,
                predicted_center_y + height / 2,
            ]
            predicted_bbox = [int(x) for x in predicted_bbox] # Ensure integer coordinates
            predicted_tracks.append({"id": player_id, "bbox": predicted_bbox, "data": player_data})

        # 2. Build cost matrix for active tracks and current detections
        num_tracks = len(predicted_tracks)
        num_detections = len(detections)
        
        # Handle cases where either tracks or detections are empty
        if num_tracks == 0 and num_detections == 0:
            self.active_players = {}
            return self.active_players
        elif num_tracks == 0:
            # All current detections are new players
            updated_active_players = {}
            for j, det in enumerate(detections):
                new_kf = KalmanFilter()
                new_kf.setup(det["bbox"])
                updated_active_players[self.next_player_id] = {
                    "bbox": det["bbox"],
                    "last_seen": frame_idx,
                    "color": (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)),
                    "kf": new_kf,
                    "features": self._extract_features(frame_img, det["bbox"])
                }
                self.next_player_id += 1
            self.active_players = updated_active_players
            return self.active_players
        elif num_detections == 0:
            # All active players are lost
            for player_id in active_player_ids:
                player_data = self.active_players[player_id]
                player_data["last_seen"] = frame_idx
                self.lost_players[player_id] = player_data
            self.active_players = {}
            return self.active_players

        # Initialize cost matrix with a high value (e.g., 1.0 for IoU cost, as 1 - 0 = 1)
        # This high value acts as a penalty for non-matches, making them less desirable than any valid match
        # If a row/column remains entirely at this high value, it means no feasible match was found
        cost_matrix = np.full((num_tracks, num_detections), 1.0) # Max cost for 1 - IoU is 1.0

        for i, track in enumerate(predicted_tracks):
            for j, det in enumerate(detections):
                iou_score = self._iou(track["bbox"], det["bbox"])
                # Only consider if there's any overlap and IoU is above a minimal threshold
                if iou_score > 0.001: # Small threshold to avoid division by zero or near-zero IoU
                    # Cost is 1 - IoU (lower cost for higher IoU)
                    cost_matrix[i, j] = 1 - iou_score

        # 3. Solve assignment problem using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 4. Update active players and identify unmatched detections/tracks
        matched_detection_indices = set()
        updated_active_players = {}

        # Process assigned matches
        for i, j in zip(row_ind, col_ind):
            # Check if the assigned cost is within the acceptable threshold
            # If cost_matrix[i, j] is 1.0, it means no good IoU match was found for this pair
            if cost_matrix[i, j] < self.cost_threshold: 
                player_id = predicted_tracks[i]["id"]
                player_data = predicted_tracks[i]["data"]
                
                updated_active_players[player_id] = {
                    "bbox": detections[j]["bbox"],
                    "last_seen": frame_idx,
                    "color": player_data["color"],
                    "kf": player_data["kf"],
                    "features": self._extract_features(frame_img, detections[j]["bbox"])
                }
                matched_detection_indices.add(j)
            else:
                # Unmatched due to high cost, move track to lost
                player_id = predicted_tracks[i]["id"]
                player_data = predicted_tracks[i]["data"]
                player_data["last_seen"] = frame_idx
                self.lost_players[player_id] = player_data

        # Identify unmatched active tracks (those not in row_ind or whose assigned cost was too high)
        for player_id in active_player_ids:
            if player_id not in updated_active_players: # If not successfully matched and updated
                # Ensure it's not already moved to lost_players in the loop above
                if player_id not in self.lost_players:
                    player_data = self.active_players[player_id]
                    player_data["last_seen"] = frame_idx
                    self.lost_players[player_id] = player_data

        # 5. Process unmatched detections (new players or re-identified lost players)
        for j, det in enumerate(detections):
            if j not in matched_detection_indices:
                re_identified = False
                # Try to re-identify from lost players using a combination of IoU and features
                # Sort lost players by how recently they were seen (most recent first)
                sorted_lost_players = sorted(self.lost_players.items(), key=lambda item: item[1]["last_seen"], reverse=True)

                for lost_id, lost_data in sorted_lost_players:
                    # Clean up old lost players here to avoid iterating over too many
                    if frame_idx - lost_data["last_seen"] > self.max_lost_frames:
                        del self.lost_players[lost_id]
                        continue

                    iou_score = self._iou(lost_data["bbox"], det["bbox"])
                    feature_similarity = self._compare_features(lost_data["features"], self._extract_features(frame_img, det["bbox"]))

                    # A more robust re-identification condition
                    # Requires a reasonable IoU and good feature similarity
                    if iou_score > 0.05 and feature_similarity > 0.75: # Adjusted thresholds
                        updated_active_players[lost_id] = {
                            "bbox": det["bbox"],
                            "last_seen": frame_idx,
                            "color": lost_data["color"],
                            "kf": lost_data["kf"],
                            "features": self._extract_features(frame_img, det["bbox"])
                        }
                        del self.lost_players[lost_id] # Remove from lost
                        re_identified = True
                        break
                
                if not re_identified:
                    # New player
                    new_kf = KalmanFilter()
                    new_kf.setup(det["bbox"])
                    updated_active_players[self.next_player_id] = {
                        "bbox": det["bbox"],
                        "last_seen": frame_idx,
                        "color": (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)),
                        "kf": new_kf,
                        "features": self._extract_features(frame_img, det["bbox"])
                    }
                    self.next_player_id += 1
        
        self.active_players = updated_active_players

        # Final clean up of old lost players (redundant but safe)
        for lost_id in list(self.lost_players.keys()):
            if frame_idx - self.lost_players[lost_id]["last_seen"] > self.max_lost_frames:
                del self.lost_players[lost_id]

        return self.active_players

# --- Main Script --- 
model = YOLO("best.pt")

video_path = "15sec_input_720p.mp4"
output_path = "output_reid_video_corrected_v5.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

tracker = PlayerTracker(max_lost_frames=90, iou_threshold=0.2, cost_threshold=0.8) # Adjusted thresholds

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    current_detections = []
    for r in results:
        for *xyxy, conf, cls in r.boxes.data:
            class_name = model.names[int(cls)]
            # Filter out detections with low confidence or non-player classes
            if class_name == 'player' and conf > 0.5: # Increased confidence threshold
                x1, y1, x2, y2 = map(int, xyxy)
                current_detections.append({'bbox': [x1, y1, x2, y2], 'conf': float(conf)})

    active_players_data = tracker.update(current_detections, frame_idx, frame)

    for player_id, player_data in active_players_data.items():
        x1, y1, x2, y2 = player_data['bbox']
        color = player_data['color']
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {player_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processed video saved to {output_path}")

