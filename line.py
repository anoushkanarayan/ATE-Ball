import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import threading

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set resolution explicitly
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Verify if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera initialized successfully.")

# Create a figure and axis for animation
fig, ax = plt.subplots()
ax.set_axis_off()
fig.patch.set_facecolor('black')  # Background color for projection
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

# Create an initial white line
line, = ax.plot([-0.5, 0.5], [0, 0], color='white', linewidth=5)

# Global variables to store target angle and line length
target_angle = 0
line_length = 0.5  # Initial line length

# Function to detect pool stick angle and distance
def detect_stick_angle():
    global target_angle, line_length
    frame_center = (cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # Flip horizontally for a mirror view
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # Edge detection
        
        # Hough Line Transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            # Find the longest line (assumed to be the pool stick)
            longest_line = max(lines, key=lambda line: np.sqrt((line[0][0] - line[0][2])**2 + (line[0][1] - line[0][3])**2))
            x1, y1, x2, y2 = longest_line[0]
            
            # Draw the line on the frame for visual reference
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Calculate the angle of the line in radians
            delta_x = x2 - x1
            delta_y = y2 - y1
            target_angle = np.arctan2(delta_y, delta_x)
            
            # Calculate the distance from the center of the frame to the midpoint of the line
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            distance = np.sqrt((mid_x - frame_center[0]) ** 2 + (mid_y - frame_center[1]) ** 2)

            # Normalize and set the line length within a limited range
            max_distance = frame.shape[1] / 2
            line_length = 0.2 + 0.8 * min(distance / max_distance, 1)

        # Display the webcam feed with the detected line
        cv2.imshow("Webcam Feed", frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to animate and rotate the line to match the pool stick angle and length
def animate(i):
    global target_angle, line_length
    # Invert the angle to make it rotate the opposite way
    angle = -target_angle
    
    # Calculate new line coordinates with rotation and dynamic length
    x = np.array([-line_length, line_length])
    y = np.array([0, 0])
    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)
    
    # Update line data
    line.set_data(x_rot, y_rot)
    return line,

# Start webcam in a separate thread
webcam_thread = threading.Thread(target=detect_stick_angle)
webcam_thread.start()

# Create an animation that rotates the line to follow the pool stick angle and adjusts length
ani = FuncAnimation(fig, animate, frames=np.arange(0, 360, 2), interval=50)

# Run plt.show() in non-blocking mode to allow simultaneous webcam and animation display
plt.show(block=False)

# Keep updating the plot and webcam feed together
try:
    while webcam_thread.is_alive():
        plt.pause(0.01)  # Keeps the Matplotlib GUI responsive
except KeyboardInterrupt:
    print("Program interrupted!")

# Make sure to close the webcam thread after use
webcam_thread.join()
