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
LOCKED_TABLE_CENTER = None
CENTER_DETECTION_TIME = None
CENTER_TIMEOUT = None  # Global timeout for center display
CENTER_MODE_ACTIVE = True  # Start in center detection mode
TRACKING_MODE_COOLDOWN = None  # Cooldown period before switching back to center mode

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
        self.table_center = None  # Add table center tracking
        self.running = True
        self.window_name = "Projection View"
        self.display_thread = None  # Initialize to None

    def update_lines(self, lines):
        with self.lock:
            self.detected_lines = lines.copy() if lines else []
            self.has_lines = len(self.detected_lines) > 0

    def update_table_center(self, center_point):
        """Update the table center point for projection"""
        with self.lock:
            self.table_center = center_point

    def should_display_center(self):
        """Check if the center should still be displayed based on timeout"""
        global CENTER_TIMEOUT, CENTER_MODE_ACTIVE
        
        if not CENTER_MODE_ACTIVE:
            return False
            
        if CENTER_TIMEOUT is None:
            return self.table_center is not None
        
        current_time = time.time()
        if current_time > CENTER_TIMEOUT:
            with self.lock:
                self.table_center = None
            return False
        
        return self.table_center is not None

    def update_plot(self, frame_num):
        for line in self.lines:
            if line in self.ax.lines:
                self.ax.lines.remove(line)
        self.lines = []
        
        with self.lock:
            current_lines = self.detected_lines.copy()
            has_lines = self.has_lines
            center = self.table_center if self.should_display_center() else None
        
        # Always draw lines if we have them
        if has_lines:
            for x1, y1, x2, y2 in current_lines:
                line, = self.ax.plot([x1, x2], [y1, y2], color='white', linewidth=3)
                self.lines.append(line)
        
        # Draw center if it's still within timeout
        if center and self.should_display_center():
            # Draw X at center
            size = 20  # Size of the X
            center_x, center_y = center
            line1, = self.ax.plot([center_x - size, center_x + size], 
                                 [center_y - size, center_y + size], 
                                 color='white', linewidth=3)
            line2, = self.ax.plot([center_x - size, center_x + size], 
                                 [center_y + size, center_y - size], 
                                 color='white', linewidth=3)
            self.lines.append(line1)
            self.lines.append(line2)
        
        return self.lines

    def create_projection_frame(self):
        frame = np.zeros((self.table_height, self.table_width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (self.table_width, self.table_height), (0, 100, 0), 2)
        
        with self.lock:
            lines = self.detected_lines.copy()
            has_lines = self.has_lines
            center = self.table_center if self.should_display_center() else None
        
        # Always draw lines if we have them
        if has_lines:
            for x1, y1, x2, y2 in lines:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 5)
        
        # Draw center if it's still within timeout
        if center and self.should_display_center():
            # Draw X at center
            center_x, center_y = center
            size = 20  # Size of the X
            # Draw X using two diagonal lines
            cv2.line(frame, (center_x - size, center_y - size), 
                    (center_x + size, center_y + size), (255, 255, 255), 5)
            cv2.line(frame, (center_x - size, center_y + size), 
                    (center_x + size, center_y - size), (255, 255, 255), 5)
        
        return cv2.rotate(frame, cv2.ROTATE_180)

    def run_opencv_display(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.table_width, self.table_height)
        while self.running:
            frame = self.create_projection_frame()
            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
            time.sleep(0.03)  # Small delay to reduce CPU usage
        cv2.destroyWindow(self.window_name)

    def run(self):
        # Only start the thread if it's not already running
        if self.display_thread is None or not self.display_thread.is_alive():
            self.display_thread = threading.Thread(target=self.run_opencv_display)
            self.display_thread.daemon = True
            self.display_thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'display_thread') and self.display_thread is not None and self.display_thread.is_alive():
            self.display_thread.join(timeout=1.0)
            self.display_thread = None  # Reset thread reference
        if hasattr(self, 'fig') and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
        cv2.destroyAllWindows()  # Close all windows to be safe

    def __del__(self):
        self.stop()

# Function to calculate the center of the table from marker data
def calculate_table_center(marker_data):
    """
    Calculate the center of the pool table using only ArUco markers.
    
    Args:
        marker_data: List of dictionaries containing marker information
        
    Returns:
        Tuple (center_x, center_y) or None if can't calculate
    """
    global LOCKED_TABLE_CENTER, CENTER_DETECTION_TIME, CENTER_TIMEOUT, CENTER_MODE_ACTIVE, TRACKING_MODE_COOLDOWN
    
    # If we're not in center mode, don't calculate a center
    if not CENTER_MODE_ACTIVE:
        return None
    
    # Check if the center display has timed out
    current_time = time.time()
    if CENTER_TIMEOUT is not None and current_time > CENTER_TIMEOUT:
        # Reset everything after timeout and switch to tracking mode
        LOCKED_TABLE_CENTER = None
        CENTER_DETECTION_TIME = None
        CENTER_TIMEOUT = None
        CENTER_MODE_ACTIVE = False
        # Set a long cooldown before switching back to center mode (30 minutes)
        TRACKING_MODE_COOLDOWN = current_time + 1800.0
        print("Center display timed out, switching to tracking mode")
        return None
    
    # If we already have a locked center, use it
    if LOCKED_TABLE_CENTER is not None:
        return LOCKED_TABLE_CENTER
        
    # Require at least 2 markers for calculation
    if not marker_data or len(marker_data) < 2:
        return None
    
    # Extract all marker centers - only use ArUco markers
    centers = [marker['center'] for marker in marker_data]
    
    # Calculate the average of all marker centers WITH the offsets
    center_x = int(sum(c[0] for c in centers) / len(centers)) + 30
    center_y = int(sum(c[1] for c in centers) / len(centers)) - 85
    
    calculated_center = (center_x, center_y)
    
    # Start or continue the center detection time
    if CENTER_DETECTION_TIME is None:
        CENTER_DETECTION_TIME = current_time
        return calculated_center
        
    # Check if we've been detecting the center for more than 5 seconds
    elif current_time - CENTER_DETECTION_TIME > 5.0:
        # Lock in the center and set timeout
        print(f"Locking table center at {calculated_center} for 10 seconds")
        LOCKED_TABLE_CENTER = calculated_center
        CENTER_TIMEOUT = current_time + 10.0  # 10 second timeout
        
    return calculated_center

# === Frame Processing ===
def process_frame(frame, detector, projection_system=None):
    global CENTER_MODE_ACTIVE, TRACKING_MODE_COOLDOWN
    
    # Check if we should switch back to center mode after cooldown
    current_time = time.time()
    if not CENTER_MODE_ACTIVE and TRACKING_MODE_COOLDOWN is not None and current_time > TRACKING_MODE_COOLDOWN:
        CENTER_MODE_ACTIVE = True
        TRACKING_MODE_COOLDOWN = None
        print("Cooldown complete, switching back to center detection mode")
    
    # Make a copy of the frame to avoid modification issues
    frame_copy = frame.copy()
    
    # Apply rotation (needed for both camera and video)
    #frame_copy = cv2.rotate(frame_copy, cv2.ROTATE_180)
    
    # Process the frame
    frame_copy, aruco_mask, bounds, table_mask, marker_data = detect_aruco_markers(frame_copy, detector, 0)
    
    # Calculate table center based on marker positions only if in center mode
    table_center = calculate_table_center(marker_data) if CENTER_MODE_ACTIVE else None
    
    # Only display table center on camera view and update projection if not timed out
    if table_center and projection_system and CENTER_MODE_ACTIVE:
        projection_system.update_table_center(table_center)
        
        # Visualize center on camera feed
        cv2.drawMarker(frame_copy, table_center, (255, 255, 255), cv2.MARKER_CROSS, 30, 3)
        cv2.putText(frame_copy, "Table Center", (table_center[0] + 15, table_center[1] - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    elif not CENTER_MODE_ACTIVE and projection_system:
        # If in tracking mode, clear the center in projection system
        projection_system.update_table_center(None)
    
    cue_ball_data, confidence = detect_white_ball(frame_copy, aruco_mask, table_mask, bounds, 0)

    cue_ball = cue_ball_data[0] if cue_ball_data else None
    cue_radius = cue_ball_data[1] if cue_ball_data else 15

    colored_balls = detect_colored_balls(frame_copy, aruco_mask, table_mask, bounds, cue_ball)
    all_detected_balls = []
    for ball_center, ball_radius, ball_color in colored_balls:
        cv2.circle(frame_copy, ball_center, ball_radius, (0, 0, 255), 2)
        color_name = get_color_name(ball_color)
        all_detected_balls.append((ball_center, ball_radius, color_name))

    trajectories = None
    if cue_ball and not CENTER_MODE_ACTIVE:  # Only calculate trajectories when not in center mode
        cv2.circle(frame_copy, cue_ball, cue_radius, (0, 255, 0), 2)
        cv2.putText(frame_copy, "Cue Ball", (cue_ball[0] - 30, cue_ball[1] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cue_line, is_pointing = detect_cue_orientation(frame_copy, cue_ball, aruco_mask, table_mask)
        if cue_line and is_pointing:
            vx, vy, _, _ = cue_line
            target_ball = find_target_ball(cue_ball, cue_line, all_detected_balls)
            if target_ball:
                center, radius, _ = target_ball
                trajectories = calculate_advanced_collision(cue_ball, center, cue_radius, radius, (vx, vy))
                
                # Debug output to verify trajectory calculation
                print(f"Calculated trajectories: {trajectories is not None}")

    # Always update projection lines if system exists, even with empty trajectories
    if projection_system and not CENTER_MODE_ACTIVE:  # Only send trajectory lines when not in center mode
        lines = extract_lines_from_trajectories(trajectories) if trajectories else []
        projection_system.update_lines(lines)
    elif projection_system and CENTER_MODE_ACTIVE:
        # Clear lines when in center mode
        projection_system.update_lines([])

    # Add mode information to the frame
    mode_text = "CENTER MODE" if CENTER_MODE_ACTIVE else "TRACKING MODE"
    cv2.putText(frame_copy, mode_text, (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
    
    # Add timeout information to the frame
    if CENTER_TIMEOUT is not None and CENTER_MODE_ACTIVE:
        time_left = max(0, int(CENTER_TIMEOUT - current_time))
        cv2.putText(frame_copy, f"Center timeout: {time_left}s", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
    elif TRACKING_MODE_COOLDOWN is not None and not CENTER_MODE_ACTIVE:
        # Show when we'll switch back to center mode (in minutes)
        minutes_left = max(0, int((TRACKING_MODE_COOLDOWN - current_time) / 60))
        cv2.putText(frame_copy, f"Center mode in: {minutes_left}m", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

    return frame_copy

# === Main ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--video', type=str, help='Path to video file instead of camera')
    parser.add_argument('--projection', action='store_true', help='Enable line projection')
    parser.add_argument('--center-time', type=int, default=10, help='Center display time in seconds (default: 10)')
    parser.add_argument('--no-center', action='store_true', help='Skip center detection and go straight to tracking mode')
    args = parser.parse_args()

    # If --no-center flag is set, start in tracking mode
    global CENTER_MODE_ACTIVE
    if args.no_center:
        CENTER_MODE_ACTIVE = False
        print("Starting directly in tracking mode (center detection disabled)")

    # Determine if we're using a camera or video file
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: Could not open video file {args.video}")
            return
    else:
        cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

    # Get frame dimensions
    table_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    table_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Table dimensions: {table_width}x{table_height}")

    # Always create projection system if --projection is specified
    projection_system = None
    if args.projection:
        projection_system = LineProjectionSystem((table_width, table_height))
        projection_system.run()

    # Set up ArUco detector
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Camera feed window
    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Feed", table_width, table_height)

    # Add playback controls for video files
    paused = False
    while True:
        if not paused or args.video is None:
            ret, frame = cap.read()
            if not ret:
                # If video file ends, loop back to beginning
                if args.video:
                    print("Restarting video from beginning")
                    cap = cv2.VideoCapture(args.video)
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to restart video")
                        break
                else:
                    print("Camera disconnected")
                    break

        # Process the frame
        processed_frame = process_frame(frame, detector, projection_system)
        
        # Display the processed frame
        cv2.imshow("Camera Feed", processed_frame)
        
        # If we're using a projection system but the separate window isn't showing,
        # make sure it's running (could have been closed by user)
        if projection_system:
            projection_system.run()  # This will check if thread exists
        
        # Handle keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit requested")
            break
        elif key == ord('c'):  # Toggle between center and tracking mode
            CENTER_MODE_ACTIVE = not CENTER_MODE_ACTIVE
            print(f"Switched to {'CENTER' if CENTER_MODE_ACTIVE else 'TRACKING'} mode")
        elif key == ord(' ') and args.video:  # Spacebar to pause/play
            paused = not paused
            print(f"Video {'paused' if paused else 'playing'}")
        elif key == ord('n') and args.video and paused:  # Next frame when paused
            ret, frame = cap.read()
            if not ret:
                cap = cv2.VideoCapture(args.video)
                ret, frame = cap.read()
                if not ret:
                    break
            print("Next frame")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if projection_system:
        projection_system.stop()

if __name__ == "__main__":
    main()