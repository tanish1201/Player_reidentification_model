import cv2
import numpy as np

# Video properties
width, height = 1280, 720
fps = 30
duration = 15 # seconds

# Define the codec and create VideoWriter object
output_path = "15sec_input_720p.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

for i in range(fps * duration):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Add some simple animation (e.g., a moving square)
    size = 100
    x = (i * 5) % (width - size)
    y = (i * 3) % (height - size)
    cv2.rectangle(frame, (x, y), (x + size, y + size), (0, 255, 0), -1)
    cv2.putText(frame, f"Frame: {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    out.write(frame)

out.release()
cv2.destroyAllWindows()
print(f"Dummy video created at {output_path}")

