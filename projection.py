import cv2
import numpy as np
import argparse
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from helpers import (
    detect_aruco_markers, detect_white_ball, update_shared_cue_ball, get_best_cue_ball,
    detect_colored_balls, get_color_name, detect_cue_orientation, find_target_ball,
    calculate_advanced_collision, extract_lines_from_trajectories
)

LOCKED_BOUNDS = [None]
LOCKED_TABLE_MASK = [None]
SHARED_CUE_BALL = {
    'positions': [None],
    'radii': [None],
    'confidences': [0.0],
    'timestamp': time.time()
}
SHARED_MARKERS = []

# === LineProjectionSystem ===
class LineProjectionSystem:
    def __init__(self, table_dimensions=(1280, 640)):
        self.table_width, self.table_height = table_dimensions
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.ax.set_facecolor('black')
        self.fig.patch.set_facecolor('black')
        self.ax.set_xlim(0, self.table_width)
        self.ax.set_ylim(self.table_height, 0)
        self.ax.set_axis_off()
        self.table_rect = plt.Rectangle((0, 0), self.table_width, self.table_height, fill=False, edgecolor='forestgreen', linewidth=2)
        self.ax.add_patch(self.table_rect)
        self.lines = []
        self.lock = threading.Lock()
        self.detected_lines = []
        self.has_lines = False
        self.running = True
        self.window_name = "Projection View"

    def update_lines(self, lines):
        with self.lock:
            self.detected_lines = lines.copy() if lines else []
            self.has_lines = len(self.detected_lines) > 0

    def update_plot(self, frame_num):
        for line in self.lines:
            if line in self.ax.lines:
                self.ax.lines.remove(line)
        self.lines = []
        with self.lock:
            current_lines = self.detected_lines.copy()
            has_lines = self.has_lines
        if has_lines:
            for x1, y1, x2, y2 in current_lines:
                line, = self.ax.plot([x1, x2], [y1, y2], color='white', linewidth=3)
                self.lines.append(line)
        return self.lines

    def create_projection_frame(self):
        frame = np.zeros((self.table_height, self.table_width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (self.table_width, self.table_height), (0, 100, 0), 2)
        with self.lock:
            lines = self.detected_lines.copy()
            has_lines = self.has_lines
        if has_lines:
            for x1, y1, x2, y2 in lines:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 5)
        #return cv2.rotate(frame, cv2.ROTATE_180)

    def run_opencv_display(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.table_width, self.table_height)
        while self.running:
            frame = self.create_projection_frame()
            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
            time.sleep(0.03)
        cv2.destroyWindow(self.window_name)

    def run(self):
        self.display_thread = threading.Thread(target=self.run_opencv_display)
        self.display_thread.daemon = True
        self.display_thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'display_thread') and self.display_thread.is_alive():
            self.display_thread.join(timeout=1.0)
        if hasattr(self, 'fig') and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
        cv2.destroyWindow(self.window_name)

    def __del__(self):
        self.stop()

# === Frame Processing ===
def process_frame(frame, detector, projection_system=None):
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame, aruco_mask, bounds, table_mask, marker_data = detect_aruco_markers(frame, detector, 0)
    cue_ball_data, confidence = detect_white_ball(frame, aruco_mask, table_mask, bounds, 0)

    cue_ball = cue_ball_data[0] if cue_ball_data else None
    cue_radius = cue_ball_data[1] if cue_ball_data else 15

    colored_balls = detect_colored_balls(frame, aruco_mask, table_mask, bounds, cue_ball)
    all_detected_balls = []
    for ball_center, ball_radius, ball_color in colored_balls:
        cv2.circle(frame, ball_center, ball_radius, (0, 0, 255), 2)
        color_name = get_color_name(ball_color)
        all_detected_balls.append((ball_center, ball_radius, color_name))

    trajectories = None
    if cue_ball:
        cv2.circle(frame, cue_ball, cue_radius, (0, 255, 0), 2)
        cv2.putText(frame, "Cue Ball", (cue_ball[0] - 30, cue_ball[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cue_line, is_pointing = detect_cue_orientation(frame, cue_ball, aruco_mask, table_mask)
        if cue_line and is_pointing:
            vx, vy, _, _ = cue_line
            target_ball = find_target_ball(cue_ball, cue_line, all_detected_balls)
            if target_ball:
                center, radius, _ = target_ball
                trajectories = calculate_advanced_collision(cue_ball, center, cue_radius, radius, (vx, vy))

    if projection_system and trajectories:
        lines = extract_lines_from_trajectories(trajectories)
        projection_system.update_lines(lines)

    return frame

# === Main ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--projection', action='store_true', help='Enable line projection')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    table_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    table_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    projection_system = LineProjectionSystem((table_width, table_height)) if args.projection else None
    if projection_system:
        projection_system.run()

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Feed", table_width, table_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame, detector, projection_system)
        cv2.imshow("Camera Feed", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if projection_system:
        projection_system.stop()

if __name__ == "__main__":
    main()
