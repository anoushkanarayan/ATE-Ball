import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import threading
import tkinter as tk

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera initialized successfully.")

# Create Matplotlib figure and axis
fig, ax = plt.subplots()
ax.set_axis_off()
fig.patch.set_facecolor('black')  # Background color for projector
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
line, = ax.plot([-0.5, 0.5], [0, 0], color='white', linewidth=5)

# Shared variables
target_angle = 0
line_length = 0.5
line_position = (0, 0)
lock = threading.Lock()

def detect_stick_angle():
    global target_angle, line_length, line_position
    frame_center = (cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

        if lines is not None:
            longest_line = max(lines, key=lambda line: np.hypot(line[0][0] - line[0][2], line[0][1] - line[0][3]))
            x1, y1, x2, y2 = longest_line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            with lock:
                target_angle = np.arctan2(y2 - y1, x2 - x1)
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                distance = np.hypot(mid_x - frame_center[0], mid_y - frame_center[1])
                max_distance = frame_width / 2
                line_length = 0.2 + 0.8 * min(distance / max_distance, 1)

                # Normalize position to [-1, 1]
                line_position = ((mid_x - frame_center[0]) / (frame_width / 2),
                                 -(mid_y - frame_center[1]) / (frame_height / 2))

        cv2.imshow("Webcam Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def animate(i):
    global target_angle, line_length, line_position

    with lock:
        angle = -target_angle
        length = line_length
        pos_x, pos_y = line_position

    x = np.array([-length, length])
    y = np.array([0, 0])
    x_rot = x * np.cos(angle) - y * np.sin(angle) + pos_x
    y_rot = x * np.sin(angle) + y * np.cos(angle) + pos_y

    line.set_data(x_rot, y_rot)
    return line,

# Start webcam thread
webcam_thread = threading.Thread(target=detect_stick_angle)
webcam_thread.start()

# Attempt to move to second screen (projector) and go fullscreen
try:
    root = tk.Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry(f"{screen_width}x{screen_height}+{screen_width}+0")  # Move to second screen
    plt.get_current_fig_manager().full_screen_toggle()  # Toggle fullscreen
except Exception as e:
    print("Fullscreen or screen move failed:", e)

# Animate and display
ani = FuncAnimation(fig, animate, interval=50)
plt.show()

# Wait for webcam thread to finish
webcam_thread.join()
