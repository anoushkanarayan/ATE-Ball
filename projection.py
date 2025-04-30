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
from line import LineProjectionSystem

LOCKED_BOUNDS = [None]
LOCKED_TABLE_MASK = [None]
SHARED_CUE_BALL = {
    'positions': [None, None],  # One position per camera
    'radii': [None, None],      # One radius per camera
    'confidences': [0.0, 0.0],  # One confidence value per camera
    'timestamp': time.time()
}
SHARED_MARKERS = []

# === Frame Processing ===
def process_frame(frame, detector, projection_system=None):
    # Make a copy of the frame to avoid modification issues
    frame_copy = frame.copy()
    
    # Apply rotation (needed for both camera and video)
    #frame_copy = cv2.rotate(frame_copy, cv2.ROTATE_180)
    
    # Process the frame
    frame_copy, aruco_mask, bounds, table_mask, marker_data = detect_aruco_markers(frame_copy, detector, 0)
    
    # ADDED: Apply adaptive brightness adjustment for low light
    # Create a grayscale copy for brightness analysis
    gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    
    # Detect white ball with enhanced parameters
    cue_ball_data, confidence = detect_white_ball(frame_copy, aruco_mask, table_mask, bounds, 0)

    # Extract cue ball information if available
    cue_ball = cue_ball_data[0] if cue_ball_data else None
    cue_radius = cue_ball_data[1] if cue_ball_data else 15
    
    # ADDED: Display current light level for debugging
    '''cv2.putText(frame_copy, f"Light: {avg_brightness:.1f}", (20, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame_copy, f"Ball Conf: {confidence:.2f}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)'''

    # Detect colored balls with the same enhanced parameters
    colored_balls = detect_colored_balls(frame_copy, aruco_mask, table_mask, bounds, cue_ball)
    all_detected_balls = []
    for ball_center, ball_radius, ball_color in colored_balls:
        cv2.circle(frame_copy, ball_center, ball_radius, (0, 0, 255), 2)
        color_name = get_color_name(ball_color)
        all_detected_balls.append((ball_center, ball_radius, color_name))

    trajectories = None
    cue_line_drawn = False  # Flag to track if we've drawn a cue line
    
    # MODIFIED: More robust cue detection and visualization
    if cue_ball:
        # Only draw the cue ball if not already drawn by the detector
        if not cue_line_drawn:
            cv2.circle(frame_copy, cue_ball, cue_radius, (0, 255, 0), 2)
            cv2.putText(frame_copy, "Cue Ball", (cue_ball[0] - 30, cue_ball[1] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # MODIFIED: Improved cue detection for low light
        # Use a wider detection area around the cue ball
        cue_line, is_pointing = detect_cue_orientation(frame_copy, cue_ball, aruco_mask, table_mask)
        
        if cue_line:
            cue_line_drawn = True
            vx, vy, _, _ = cue_line
            
            # ADDED: Show detected cue direction angle
            angle_deg = np.degrees(np.arctan2(vy, vx))
            '''cv2.putText(frame_copy, f"Angle: {angle_deg:.1f}°", 
                       (cue_ball[0] - 30, cue_ball[1] + cue_radius + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)'''
            
            if is_pointing:
                # Find target ball with improved parameters
                target_ball = find_target_ball(cue_ball, cue_line, all_detected_balls)
                
                if target_ball:
                    center, radius, _ = target_ball
                    # MODIFIED: Make trajectory more visible in low light
                    trajectories = calculate_advanced_collision(cue_ball, center, cue_radius, radius, (vx, vy))
                    
                    # ADDED: Highlight the target ball
                    cv2.circle(frame_copy, center, radius, (0, 255, 255), 3)  # Thicker circle
                    cv2.putText(frame_copy, "Target", (center[0] - 25, center[1] - radius - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # Debug output to verify trajectory calculation
                    print(f"Calculated trajectories: {trajectories is not None}")
                else:
                    # No target ball found, but cue is pointing at the cue ball
                    # Create a simple straight line trajectory with clear visibility
                    extension = 500  # Length of the projected line
                    end_x = int(cue_ball[0] + vx * extension)
                    end_y = int(cue_ball[1] + vy * extension)
                    
                    # Create a simple trajectory dictionary with just a straight line
                    trajectories = {
                        'will_collide': False,
                        'cue_initial': (int(cue_ball[0]), int(cue_ball[1])),
                        'cue_path': [(int(cue_ball[0]), int(cue_ball[1])), (end_x, end_y)]
                    }
                    
                    # Draw the straight line on the frame with high visibility
                    cv2.line(frame_copy, cue_ball, (end_x, end_y), (0, 255, 255), 3)  # Thicker line
                    cv2.putText(frame_copy, "Projected Path", (end_x - 50, end_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Always update projection lines if system exists, even with empty trajectories
    if projection_system:
        lines = extract_lines_from_trajectories(trajectories) if trajectories else []
        
        # Debug output
        # print(f"Extracted lines: {len(lines) if lines else 0}")
        
        projection_system.update_lines(lines)
        
        # Draw projection lines directly on the camera feed with enhanced visibility
        current_lines, has_lines = projection_system.get_current_lines()
        if has_lines:
            for x1, y1, x2, y2 in current_lines:
                cv2.line(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 4)  # Brighter, thicker lines

    return frame_copy

# === Main ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--video', type=str, help='Path to video file instead of camera')
    parser.add_argument('--projection', action='store_true', help='Enable line projection')
    args = parser.parse_args()

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

        # TOGGLE THIS ON AND OFF FOR DEBUGGING PURPOSES
        #processed_frame = cv2.rotate(processed_frame, cv2.ROTATE_180)
        
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