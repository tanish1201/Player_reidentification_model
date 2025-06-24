# Player Re-Identification Project: Execution Instructions

This document provides concise instructions on how to set up and run the player re-identification code.

## Setup and Installation

To run the project, you need to set up a Python environment and install the necessary dependencies.

1.  **Clone the repository (if applicable):**
    If you have this project in a Git repository, clone it to your local machine:
    ```bash
    git clone <repository_url>
    cd player_reid # Navigate to the project directory
    ```
    If you received the files directly, ensure all project files (scripts, `requirements.txt`, `best.pt`, `15sec_input_720p.mp4`) are in the same directory.

2.  **Create a virtual environment (recommended):**
    It's good practice to use a virtual environment to manage project dependencies.
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Install all required Python packages using `pip`. Note that `scipy` is also required for the latest version of the script.
    ```bash
    pip install -r requirements.txt scipy
    ```

4.  **Obtain the model file:**
    The object detection model, `best.pt` (a fine-tuned Ultralytics YOLOv11 model), is crucial for this project. Ensure this file is placed in the project's root directory (the same directory as the Python scripts).

5.  **Obtain the video file:**
    The input video for re-identification is `15sec_input_720p.mp4`. Ensure this video file is also placed in the project's root directory.

## How to Run the Code

After completing the setup and ensuring all necessary files (`best.pt`, `15sec_input_720p.mp4`, `reid_script_corrected_v5.py`, `requirements.txt`) are in the same directory, you can execute the main re-identification script.

Open your terminal or command prompt, navigate to the project directory, and run the following command:

```bash
python reid_script_corrected_v5.py
```

### Output

The script will process the `15sec_input_720p.mp4` video frame by frame. Upon successful completion, it will generate an output video file named `output_reid_video_corrected_v5.mp4` in the same directory. This output video will display the detected players with their assigned IDs and bounding boxes, reflecting the re-identification logic applied.

## Important Notes

*   **Computational Intensity:** Object detection and tracking are computationally intensive tasks. Running the script, especially on longer videos or without GPU acceleration, can take a significant amount of time.
*   **Model Performance:** The accuracy of player detection and the robustness of re-identification are directly influenced by the quality of the `best.pt` model and the sophistication of the tracking algorithm. While the provided script incorporates advanced data association (Hungarian algorithm) and Kalman filtering, it still relies on a basic feature comparison (average color) for re-identification, which has limitations.
*   **Error Analysis:** For a detailed understanding of the challenges and limitations encountered during the development of this project, including specific errors from v1 to v5 and areas for future improvement, please refer to the `error_analysis.md` file.

