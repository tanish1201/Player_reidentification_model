# Player Re-Identification in a Single Feed

## Objective

The primary objective of this project is to implement a real-time player re-identification and tracking solution for a 15-second video clip. The solution aims to accurately identify each player and ensure that players who temporarily go out of frame and reappear are assigned the same identity as before. This simulates a real-time re-identification and player tracking system, crucial for applications such as sports analytics.

## Introduction

This repository contains the code and documentation for a player re-identification system built using the Ultralytics YOLOv11 object detection model. The challenge involves not just detecting players in each frame, but also maintaining their unique identities across the entire video, even when they are occluded or leave and re-enter the scene. This project explores various approaches to object tracking and re-identification, highlighting the complexities and solutions for achieving robust performance in dynamic environments like sports footage.

## Model Used

The object detection component of this project utilizes a fine-tuned version of **Ultralytics YOLOv11**, provided as a `best.pt` PyTorch model file. YOLO (You Only Look Look Once) is a popular real-time object detection system known for its balance of speed and accuracy. This specific model has been trained for detecting players and the ball in sports contexts.

## Core Concepts

Implementing player re-identification involves several fundamental computer vision and machine learning concepts:

*   **Object Detection:** Identifying and localizing players (and the ball) within each video frame using the YOLOv11 model.
*   **Object Tracking:** Linking detections across consecutive frames to form continuous trajectories for each player.
*   **Re-Identification (Re-ID):** The crucial task of re-assigning the original ID to a player who has disappeared from view (e.g., due to occlusion or leaving the frame) and subsequently reappeared. This requires comparing features of newly detected objects with those of previously seen (and potentially lost) objects.
*   **Kalman Filters:** Used for predicting the future state (position and velocity) of tracked objects, helping to smooth trajectories and improve association accuracy.
*   **Data Association (Hungarian Algorithm):** An optimization algorithm used to find the optimal assignment between predicted track locations and current detections, minimizing a defined cost (e.g., based on IoU).

For a more detailed explanation of these concepts, please refer to the `reid_errors_analysis.md` document in this repository.

## Implementation Details and Evolution

The project's core logic is encapsulated in the Python scripts, which evolved through several iterations to address various challenges in player re-identification:

### Initial Approach (`reid_script.py`)

The first version of the script implemented a basic IoU-based tracking and re-identification logic. It involved:

*   Loading the YOLOv11 model.
*   Processing video frames sequentially.
*   Detecting players in each frame.
*   Assigning new IDs to unmatched detections.
*   Attempting to re-identify lost players based on IoU with a lower threshold.

**Challenges Faced:** This initial approach suffered from significant issues, including frequent ID switches, fragmented player trajectories, and a high susceptibility to occlusions, leading to unreliable tracking.

### Improved Approach with Kalman Filter and Basic Features (`reid_script_corrected.py` and `reid_script_corrected_v2.py`)

To enhance tracking robustness, a Kalman Filter was integrated into the `PlayerTracker` class to predict player positions, and a simple color-based feature (`_extract_features`) was introduced for re-identification. The `reid_script_corrected_v2.py` further refined the thresholds and `max_lost_frames`.

**Challenges Faced:** While these improvements mitigated some issues, problems like inconsistent IDs, multiple bounding boxes for a single player, and misidentification of non-player objects (e.g., goal post) persisted. The simple feature comparison proved insufficient for robust re-identification in complex scenarios, and the IoU-based matching was still prone to errors in crowded scenes.

### Advanced Data Association with Hungarian Algorithm (`reid_script_corrected_v3.py` and `reid_script_corrected_v4.py` / `reid_script_corrected_v5.py`)

The latest iterations (`reid_script_corrected_v3.py` and its subsequent minor fixes `reid_script_corrected_v4.py` and `reid_script_corrected_v5.py`) represent a significant step towards more robust tracking by incorporating the **Hungarian Algorithm** (`scipy.optimize.linear_sum_assignment`) for optimal data association. This approach aims to:

*   Find the best possible match between predicted track locations and current detections, minimizing a cost function (1 - IoU).
*   Improve ID consistency by making more informed assignment decisions.
*   Address the `ValueError: cost matrix is infeasible` by ensuring the cost matrix is always valid for the Hungarian algorithm.

**Current Limitations:** Despite these advancements, the system still faces limitations primarily due to the absence of robust, deep learning-based appearance features. The simple average color feature is not discriminative enough for truly reliable re-identification, especially after long occlusions or when players reappear in different parts of the frame. The misidentification of non-player objects (like the goal post) is also an inherent limitation of the provided YOLOv11 model itself.

## Setup and Installation

To run the project, you need to set up a Python environment and install the necessary dependencies.

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd player_reid
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt scipy
    ```

4.  **Obtain the model file:**
    Place the `best.pt` YOLOv11 model file (provided separately) into the project's root directory.

5.  **Obtain the video file:**
    Place the `15sec_input_720p.mp4` video file (your input video) into the project's root directory.

## How to Run

After setting up the environment and placing the necessary files, you can run the re-identification script:

```bash
python reid_script_corrected_v5.py
```

The script will process the `15sec_input_720p.mp4` video and generate an output video named `output_reid_video_corrected_v5.mp4` in the same directory. This output video will show the detected players with their assigned IDs and bounding boxes.

## Files in this Repository

*   `reid_script_corrected_v5.py`: The main Python script containing the latest re-identification logic.
*   `requirements.txt`: Lists the Python packages required to run the script.
*   `implementation_guide.pdf`: A detailed document explaining the core concepts and initial implementation steps of player re-identification.
*   `reid_errors_analysis.pdf`: An in-depth analysis of the persistent errors encountered during the development of the tracking logic, distinguishing between model and algorithm limitations.
*   `create_dummy_video.py`: A utility script to create a dummy video file for testing purposes (if `15sec_input_720p.mp4` is not available).
*   `best.pt`: The YOLOv11 object detection model (should be placed here).
*   `15sec_input_720p.mp4`: The input video file (should be placed here).

## Known Limitations and Future Improvements

*   **Lack of Robust Appearance Features:** The current feature extraction (average color) is very basic. Integrating a pre-trained re-identification model (e.g., a Siamese network) to extract highly discriminative appearance embeddings would significantly improve re-identification accuracy, especially for players who are occluded for longer periods or reappear far from their last known position.
*   **Full DeepSORT Implementation:** A complete implementation of DeepSORT, which combines Kalman filtering with deep appearance features and a robust association metric, would provide state-of-the-art performance.
*   **Occlusion Handling:** More sophisticated strategies for managing partial and full occlusions are needed to prevent ID switches and track fragmentation in crowded scenes.
*   **Model Refinement:** The provided `best.pt` model occasionally misidentifies non-player objects (like the goal post). Further fine-tuning or training the object detection model on a more diverse and carefully annotated dataset could mitigate this.
*   **Performance Optimization:** For real-time applications, optimizing the inference speed of the YOLO model (e.g., using TensorRT) and the tracking algorithm would be crucial.

