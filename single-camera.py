import cv2
import numpy as np
import argparse
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Global variables to track state
LOCKED_BOUNDS = None
LOCKED_TABLE_MASK = None

class LineProjectionSystem:
    def __init__(self, shared_data=None):
        """
        Initialize the line projection system.
        
        Args:
            shared_data: A dictionary to share data with the main application
        """
        # Create matplotlib figure for projection
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_facecolor('black')  # Set axis background to black
        self.fig.patch.set_facecolor('black')  # Set figure background to black
        self.ax.set_xlim(0, 640)
        self.ax.set_ylim(480, 0)  # Invert Y-axis to match OpenCV coordinates
        self.ax.set_axis_off()  # Hide axis
        
        # Initialize line collections for visualization
        self.lines = []
        
        # Thread lock for synchronization
        self.lock = threading.Lock()
        
        # Shared data with main application
        self.shared_data = shared_data if shared_data is not None else {}
        
        # Variables to store detected lines
        self.detected_lines = []  # List of (x1, y1, x2, y2) tuples
        
        # Flag to control processing loop
        self.running = True
        
        # OpenCV window name for projection view
        self.window_name = "Projection View"
        
    def update_lines(self, lines):
        """Update detected lines with thread lock."""
        with self.lock:
            self.detected_lines = lines.copy() if lines else []
            
    def update_plot(self, frame_num):
        """Update function for matplotlib animation."""
        # Clear previous lines
        for line in self.lines:
            if line in self.ax.lines:
                self.ax.lines.remove(line)
        self.lines = []
        
        # Get current detected lines with thread lock
        with self.lock:
            current_lines = self.detected_lines.copy()
            
        # Draw each line
        for x1, y1, x2, y2 in current_lines:
            line, = self.ax.plot([x1, x2], [y1, y2], color='white', linewidth=3)
            self.lines.append(line)
            
        # Update shared data if needed
        if 'frame_ready' in self.shared_data:
            self.shared_data['frame_ready'] = True
            
        return self.lines
    
    def create_projection_frame(self, frame_size=(640, 480)):
        """Create a projection frame with detected lines."""
        # Get frame size from shared data or use default
        if 'frame_size' in self.shared_data:
            frame_size = self.shared_data['frame_size']
            
        # Create a black frame
        projection_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        
        # Get current detected lines with thread lock
        with self.lock:
            current_lines = self.detected_lines.copy()
        
        # Draw each line in white
        for x1, y1, x2, y2 in current_lines:
            cv2.line(projection_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 5)
            
        return projection_frame
        
    def run_opencv_display(self):
        """Run OpenCV-based display for projection."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        while self.running:
            # Create projection frame
            projection_frame = self.create_projection_frame()
            
            # Display the frame
            cv2.imshow(self.window_name, projection_frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
                
            # Small delay to reduce CPU usage
            cv2.waitKey(30)
        
        cv2.destroyWindow(self.window_name)
        
    def run_matplotlib_display(self):
        """Run Matplotlib-based display for projection."""
        # Start matplotlib animation for projection display
        self.ani = FuncAnimation(
            self.fig, 
            self.update_plot,
            interval=50,  # Update every 50ms (20 fps)
            blit=True
        )
        
        # Show the matplotlib display (blocking call)
        plt.show()
        
    def run(self, use_matplotlib=False):
        """Run the line projection system."""
        # Start display thread
        if use_matplotlib:
            self.display_thread = threading.Thread(target=self.run_matplotlib_display)
        else:
            self.display_thread = threading.Thread(target=self.run_opencv_display)
            
        self.display_thread.daemon = True
        self.display_thread.start()
        
    def stop(self):
        """Stop the projection system."""
        self.running = False
        if hasattr(self, 'display_thread') and self.display_thread.is_alive():
            self.display_thread.join(timeout=1.0)
            
        # Close matplotlib figure if it exists
        if hasattr(self, 'fig') and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
            
        # Close OpenCV window if it exists
        cv2.destroyWindow(self.window_name)
        
    def __del__(self):
        """Clean up resources."""
        self.stop()

# Helper function to extract lines from trajectories
def extract_lines_from_trajectories(trajectories):
    """Extract line segments from trajectory data."""
    lines = []
    
    if not trajectories:
        return lines
    
    if trajectories.get('will_collide', False):
        # Cue stick to collision point
        start = trajectories['cue_path'][0]
        end = trajectories['collision_point']
        lines.append((start[0], start[1], end[0], end[1]))
        
        # Cue ball after collision
        start = trajectories['collision_point'] 
        end = trajectories['cue_path'][3]
        lines.append((start[0], start[1], end[0], end[1]))
        
        # Target ball trajectory
        start = trajectories['collision_point']
        end = trajectories['target_path'][1]
        lines.append((start[0], start[1], end[0], end[1]))
    else:
        # No collision - just the straight path
        start = trajectories['cue_path'][0]
        end = trajectories['cue_path'][1]
        lines.append((start[0], start[1], end[0], end[1]))
    
    return lines

def calculate_advanced_collision(cue_pos, target_pos, cue_radius, target_radius, stick_direction):
    """
    Calculate advanced collision trajectories between cue ball and target ball
    using the physics from the matplotlib example.
    
    Args:
        cue_pos: Position of the cue ball as (x, y)
        target_pos: Position of the target ball as (x, y)
        cue_radius: Radius of the cue ball
        target_radius: Radius of the target ball
        stick_direction: Direction vector of the cue stick (vx, vy)
        
    Returns:
        Dict with trajectory points and collision information
    """
    try:
        # Extract scalar values from stick_direction if it's an array
        vx = float(stick_direction[0]) if hasattr(stick_direction[0], '__len__') else float(stick_direction[0])
        vy = float(stick_direction[1]) if hasattr(stick_direction[1], '__len__') else float(stick_direction[1])
        
        # Convert inputs to numpy arrays
        cue = np.array([float(cue_pos[0]), float(cue_pos[1])])
        target = np.array([float(target_pos[0]), float(target_pos[1])])
        r = (float(cue_radius) + float(target_radius)) / 2  # Average radius for calculations
        stick_vec = np.array([vx, vy])
        
        # Normalize stick vector
        stick_vec = stick_vec / np.linalg.norm(stick_vec)
        
        # Calculate vector from cue to target
        d_vec = target - cue
        
        # Project d_vec onto stick_vec
        proj_d = np.dot(d_vec, stick_vec) * stick_vec
        
        # Rejection of d_vec from stick_vec (perpendicular component)
        rej_d = d_vec - proj_d
        
        # Check if collision is possible
        if np.linalg.norm(rej_d) > 2*r:
            # No collision possible - balls will miss
            # Just return a straight line trajectory
            extension = 500  # Long line
            end_point = cue + stick_vec * extension
            
            return {
                'will_collide': False,
                'cue_initial': (int(cue[0]), int(cue[1])),
                'cue_path': [(int(cue[0]), int(cue[1])), (int(end_point[0]), int(end_point[1]))],
                'target_path': []
            }
        
        # Calculate where the cue ball would be at collision
        mag_sub = np.sqrt(4*r*r - np.linalg.norm(rej_d)**2)
        new_cue = cue + proj_d * (1 - mag_sub/np.linalg.norm(proj_d))
        
        # Calculate collision point
        diff_vec = (target - new_cue) / (2*r)
        diff_vec = diff_vec / np.linalg.norm(diff_vec)  # Normalize
        collision = new_cue + diff_vec * r
        
        # Calculate velocity vectors after collision
        v_target = np.dot(stick_vec, diff_vec) * diff_vec
        v_cue = stick_vec - v_target
        
        # Scale vectors for visualization
        scale_factor = 150  # Adjust this for visualization
        v_target = v_target * scale_factor
        v_cue = v_cue * scale_factor
        
        # Calculate points for drawing
        cue_line_start = cue
        cue_line_end = new_cue
        
        # Extend the lines past the collision point
        target_line_start = collision
        target_line_end = collision + v_target
        
        cue_after_start = collision
        cue_after_end = collision + v_cue
        
        # Start drawing the cue stick line from the edge of the cue ball,
        # not from its center
        start_v_cue = r * stick_vec / np.linalg.norm(stick_vec)
        cue_stick_start = cue + start_v_cue
        
        return {
            'will_collide': True,
            'cue_initial': (int(cue[0]), int(cue[1])),
            'target_pos': (int(target[0]), int(target[1])),
            'collision_point': (int(collision[0]), int(collision[1])),
            'cue_path': [
                (int(cue_stick_start[0]), int(cue_stick_start[1])),
                (int(new_cue[0]), int(new_cue[1])),
                (int(cue_after_start[0]), int(cue_after_start[1])),
                (int(cue_after_end[0]), int(cue_after_end[1]))
            ],
            'target_path': [
                (int(target_line_start[0]), int(target_line_start[1])),
                (int(target_line_end[0]), int(target_line_end[1]))
            ]
        }
    except Exception as e:
        print(f"Error in advanced collision calculation: {e}")
        import traceback
        traceback.print_exc()
        return None

def distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def detect_aruco_markers(frame, detector):
    """Detect ArUco markers in the frame and return the frame with markers highlighted,
    an ArUco mask, table bounds, and table mask."""
    global LOCKED_BOUNDS, LOCKED_TABLE_MASK

    if LOCKED_BOUNDS is not None and LOCKED_TABLE_MASK is not None:
        return frame, np.ones(frame.shape[:2], dtype=np.uint8) * 255, LOCKED_BOUNDS, LOCKED_TABLE_MASK

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    markerCorners, markerIds, _ = detector.detectMarkers(gray)

    aruco_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255  # Default mask

    if markerIds is not None:
        all_corners = []
        for corners, marker_id in zip(markerCorners, markerIds):
            corners = corners.reshape((4, 2)).astype(int)
            all_corners.extend(corners)

            # Draw the ArUco markers
            cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, f"ID: {marker_id[0]}", (corners[0][0], corners[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if all_corners:
            x_min = min([c[0] for c in all_corners])
            y_min = min([c[1] for c in all_corners])
            x_max = max([c[0] for c in all_corners])
            y_max = max([c[1] for c in all_corners])
            bounds = (x_min, y_min, x_max, y_max)

            table_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            if len(all_corners) >= 4:
                hull = cv2.convexHull(np.array(all_corners))
                cv2.fillPoly(table_mask, [hull], 255)
                cv2.polylines(frame, [hull], isClosed=True, color=(0, 255, 255), thickness=1)
            else:
                table_points = np.array([[x_min, y_min], [x_max, y_min], 
                                         [x_max, y_max], [x_min, y_max]])
                cv2.fillPoly(table_mask, [table_points], 255)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1)

            # Lock in detected values
            LOCKED_BOUNDS = bounds
            LOCKED_TABLE_MASK = table_mask

            return frame, aruco_mask, bounds, table_mask

# If detection fails and bounds are locked, return stored values
    if LOCKED_BOUNDS is not None and LOCKED_TABLE_MASK is not None:
        return frame, aruco_mask, LOCKED_BOUNDS, LOCKED_TABLE_MASK
        
    # Default fallback case
    return frame, aruco_mask, (0, 0, frame.shape[1], frame.shape[0]), np.ones(frame.shape[:2], dtype=np.uint8) * 255

def detect_white_ball(frame, aruco_mask, table_mask, bounds):
    """Detect the white cue ball in the frame."""
    # Method 1: HSV color filtering
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # HSV range for white - slightly modified for better detection
    lower_white, upper_white = np.array([0, 0, 200]), np.array([180, 40, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Only consider areas within the table bounds
    mask = cv2.bitwise_and(mask, table_mask)
    
    # First try with contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Filter contours by circularity and size
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Skip tiny contours
            if area < 300:
                continue
                
            perimeter = cv2.arcLength(contour, True)
            # Avoid division by zero
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.7:  # Circular enough to be a ball
                    valid_contours.append(contour)
        
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            if radius > 10:
                return (int(x), int(y)), int(radius)
    
    return None, None

def detect_colored_balls(frame, aruco_mask, table_mask, bounds, cue_ball=None):
    """
    Detect all colored balls on the table, excluding the cue ball.
    Returns a list of tuples (center, radius, dominant_color).
    """
    # Create a copy of the frame for visualization/debugging
    debug_frame = frame.copy()
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # We'll use Hough Circles to find all balls, then filter by color
    # Convert to grayscale for Hough Circles detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply the table mask to the grayscale image
    # This restricts circle detection to just the table area
    masked_gray = cv2.bitwise_and(gray, gray, mask=table_mask)
    
    # Apply slight blur to reduce noise
    masked_gray = cv2.GaussianBlur(masked_gray, (5, 5), 0)
    
    # Use Hough Circles to detect all circular objects
    # Parameters need tuning for your specific table setup
    circles = cv2.HoughCircles(
        masked_gray,
        cv2.HOUGH_GRADIENT,
        dp=1,           # Resolution ratio
        minDist=30,     # Minimum distance between circles
        param1=50,      # Upper threshold for Canny edge detector
        param2=25,      # Threshold for center detection
        minRadius=10,   # Minimum radius to detect
        maxRadius=30    # Maximum radius to detect
    )
    
    colored_balls = []
    
    if circles is not None:
        # Convert circle parameters to integers
        circles = np.uint16(np.around(circles))
        
        # Process each detected circle
        for circle in circles[0, :]:
            x, y, radius = circle
            center = (int(x), int(y))
            
            # Skip if this is potentially the cue ball
            if cue_ball and distance(center, cue_ball) < radius * 1.5:
                continue
                
            # Additional check to ensure the ball is fully within the table bounds
            if (x - radius < bounds[0] or 
                y - radius < bounds[1] or 
                x + radius > bounds[2] or 
                y + radius > bounds[3]):
                continue
                
            # Make sure the center is within the table mask
            if table_mask[y, x] == 0:
                continue
                
            # Create a mask for this circle
            ball_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(ball_mask, center, radius, 255, -1)
            
            # Apply the table mask
            ball_mask = cv2.bitwise_and(ball_mask, table_mask)
            
            # If the masked area is too small, skip this circle
            # (helps filter out partial circles at the edge of the table)
            if np.sum(ball_mask) < np.pi * radius * radius * 0.7:
                continue
            
            # Get the average color in HSV space
            masked_hsv = cv2.bitwise_and(hsv, hsv, mask=ball_mask)
            if np.sum(ball_mask) > 0:  # Ensure we have some pixels to analyze
                hsv_values = masked_hsv[ball_mask > 0]
                mean_color = np.mean(hsv_values, axis=0).astype(int)
                
                # Check if the ball is white (cue ball)
                h, s, v = mean_color
                if s < 40 and v > 180:  # Low saturation, high value
                    continue  # Skip cue ball
                
                # Add to the list of colored balls
                colored_balls.append((center, radius, mean_color))
                
                # Draw circle on debug frame
                cv2.circle(debug_frame, center, radius, (0, 0, 255), 2)
    
    return colored_balls

def get_color_name(hsv_color):
    """Convert HSV color to a human-readable name."""
    h, s, v = hsv_color
    
    # Define color ranges
    if s < 50:
        return "Black" if v < 100 else "White"
    
    if h < 10 or h > 170:
        return "Red"
    elif 10 <= h < 25:
        return "Orange"
    elif 25 <= h < 35:
        return "Yellow"
    elif 35 <= h < 75:
        return "Green"
    elif 75 <= h < 130:
        return "Blue"
    elif 130 <= h < 170:
        return "Purple"
    
    return "Unknown"

def detect_cue_orientation(frame, cue_ball, aruco_mask, table_mask=None):
    """Detect the orientation of the cue stick."""
    if cue_ball is None:
        return None, False
        
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # HSV range for silver (light gray to white-ish colors)
    # The exact range may need fine-tuning depending on the lighting conditions.
    lower_silver = np.array([0, 0, 180])  # Light gray/white colors
    upper_silver = np.array([180, 30, 255])  # Light gray/white colors
    
    # Create mask for silver objects (cue stick in this case)
    mask = cv2.inRange(hsv, lower_silver, upper_silver)
    
    # Apply the ArUco mask to exclude those regions
    mask = cv2.bitwise_and(mask, aruco_mask)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest silver contour (the cue stick)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit a line to the silver contour (cue stick)
        [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Extract scalar values from single-element arrays
        vx_scalar = float(vx[0])
        vy_scalar = float(vy[0])
        x_scalar = float(x[0])
        y_scalar = float(y[0])
        
        # Invert direction by 180 degrees
        vx_scalar, vy_scalar = -vx_scalar, -vy_scalar
        
        # Get the cue ball position
        cue_x, cue_y = cue_ball
        
        # Calculate distance from cue line to ball center
        # The cue line starts at (x,y) and goes in direction (vx,vy)
        
        # Project vector from cue line point to ball onto the perpendicular to the cue direction
        perp_x, perp_y = -vy_scalar, vx_scalar  # Perpendicular to (vx, vy)
        vec_to_ball_x, vec_to_ball_y = cue_x - x_scalar, cue_y - y_scalar
        
        # Distance from line to point
        distance = abs(vec_to_ball_x * perp_x + vec_to_ball_y * perp_y) / np.sqrt(perp_x**2 + perp_y**2)
        
        # Check if ball is ahead of the cue, not behind
        dot_product = vec_to_ball_x * vx_scalar + vec_to_ball_y * vy_scalar
        is_ahead = dot_product > 0
        
        # Maximum allowed distance (can be adjusted based on ball radius and tolerance)
        max_distance = 20  # pixels
        
        is_pointing_at_ball = is_ahead and distance < max_distance
        
        # Return the line parameters (direction and origin) and whether it's pointing at the ball
        return (vx_scalar, vy_scalar, cue_x, cue_y), is_pointing_at_ball
        
    return None, False

def find_target_ball(cue_ball, cue_line, all_balls):
    """
    Find the ball that the cue is pointing at.
    Returns the ball information (center, radius, color) or None if no ball is targeted.
    """
    if cue_ball is None or cue_line is None or not all_balls:
        return None
    
    vx, vy, x0, y0 = cue_line
    cue_x, cue_y = cue_ball
    
    closest_ball = None
    min_ball_distance = float('inf')
    
    for ball_info in all_balls:
        center, radius, color = ball_info
        ball_x, ball_y = center
        
        # Calculate perpendicular distance from ball center to cue line
        # The line is defined by: (x0, y0) and direction (vx, vy)
        # Avoid division by zero
        if vx == 0:
            # Vertical line
            perp_distance = abs(ball_x - x0)
        else:
            # Distance from point to line formula
            perp_distance = abs(vy*(ball_x - x0) - vx*(ball_y - y0)) / np.sqrt(vx*vx + vy*vy)
        
        # Calculate distance from cue ball to this ball
        ball_distance = distance(cue_ball, center)
        
        # Check if the ball is in front of the cue ball (not behind)
        # Calculate the dot product of the vector from cue to ball and the cue direction
        ball_vector = [ball_x - cue_x, ball_y - cue_y]
        direction_vector = [vx, vy]
        dot_product = ball_vector[0]*direction_vector[0] + ball_vector[1]*direction_vector[1]
        
        # If the ball is in front of the cue and close enough to the line
        if dot_product > 0 and perp_distance < radius + 5:  # Adding a small tolerance
            if ball_distance < min_ball_distance:
                min_ball_distance = ball_distance
                closest_ball = ball_info
    
    return closest_ball

def process_frame(frame, detector, projection_system=None):
    """
    Process a frame, detect balls and trajectories, and update projection system.
    
    Args:
        frame: Input video frame
        detector: ArUco marker detector
        projection_system: Optional LineProjectionSystem instance
    
    Returns:
        Processed frame with visualization
    """
    # Make a copy of the original frame for clean visualization
    original_frame = frame.copy()
    
    # Detect ArUco markers and get boundary coordinates and table mask
    frame, aruco_mask, bounds, table_mask = detect_aruco_markers(frame, detector)
    
    # Detect all balls using the Hough Circle method - much more reliable
    gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=25,
        minRadius=10,
        maxRadius=30
    )
    
    # Detect cue ball separately
    cue_ball, cue_radius = detect_white_ball(frame, aruco_mask, table_mask, bounds)
    
    # Also detect colored balls separately (as a fallback)
    colored_balls = detect_colored_balls(frame, aruco_mask, table_mask, bounds, cue_ball)
    
    # Store detected ball information
    all_detected_balls = []
    
    # Mark all balls detected by Hough Circles (this should work better)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for circle in circles[0, :]:
            x, y, radius = circle
            center = (int(x), int(y))
            
            # Skip if this point is outside the table area
            if table_mask[y, x] == 0:
                continue
                
            # Additional check to ensure the entire ball is within the table bounds
            if (x - radius < bounds[0] or 
                y - radius < bounds[1] or 
                x + radius > bounds[2] or 
                y + radius > bounds[3]):
                continue
                
            # Check if this is the cue ball we already detected
            is_cue_ball = False
            if cue_ball and distance(center, cue_ball) < radius:
                is_cue_ball = True
                continue  # Skip cue ball, we draw it separately
                
            # Draw the circle for non-cue balls
            if not is_cue_ball:
                # Draw circle around the ball
                cv2.circle(frame, center, radius, (0, 0, 255), 2)
                
                # Determine color
                ball_mask = np.zeros(original_frame.shape[:2], dtype=np.uint8)
                cv2.circle(ball_mask, center, radius, 255, -1)
                
                hsv = cv2.cvtColor(original_frame, cv2.COLOR_BGR2HSV)
                masked_hsv = cv2.bitwise_and(hsv, hsv, mask=ball_mask)
                if np.sum(ball_mask) > 0:
                    mean_color = np.mean(masked_hsv[ball_mask > 0], axis=0).astype(int)
                    color_name = get_color_name(mean_color)
                    
                    # Store ball information
                    all_detected_balls.append((center, radius, color_name))
    
    # Draw any colored balls from our separate detection (as a backup)
    # This helps ensure we don't miss any balls
    for ball_center, ball_radius, ball_color in colored_balls:
        # Check if we already drew this ball from Hough Circles
        already_drawn = False
        if circles is not None:
            for circle in circles[0, :]:
                circle_center = (int(circle[0]), int(circle[1]))
                if distance(ball_center, circle_center) < ball_radius:
                    already_drawn = True
                    break
        
        if not already_drawn:
            # Draw circle around the ball
            cv2.circle(frame, ball_center, ball_radius, (0, 0, 255), 2)
            
            # Get color name
            color_name = get_color_name(ball_color)
            
            # Store ball information
            all_detected_balls.append((ball_center, ball_radius, color_name))
    
    # Initialize target ball and trajectories
    target_ball = None
    trajectories = None
    
    # Draw the cue ball and detect cue orientation
    if cue_ball:
        cv2.circle(frame, cue_ball, cue_radius, (0, 255, 0), 2)
        cv2.putText(frame, "Cue Ball", (cue_ball[0] - 30, cue_ball[1] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
        # Detect cue orientation
        cue_line, is_pointing_at_ball = detect_cue_orientation(frame, cue_ball, aruco_mask, table_mask)
        
        # Draw the cue line ONLY if it's pointing at the ball
        if cue_line and is_pointing_at_ball:
            vx, vy, x0, y0 = cue_line
            
            # Find the ball that the cue is pointing at (if any)
            target_ball = find_target_ball(cue_ball, cue_line, all_detected_balls)
            
            # If no target ball is found, just draw the regular cue line
            if not target_ball:
                # Calculate end point of the line (extend in cue direction)
                # Using scalar values to avoid deprecation warning
                end_point = (int(x0 + vx * 400), int(y0 + vy * 400))
                
                # Start point is the cue ball
                start_point = (int(x0), int(y0))
                
                # Draw the line
                cv2.line(frame, start_point, end_point, (255, 255, 255), 2)  # White line
    
    # If a target ball is found, label it and draw the trajectory
    if target_ball:
        center, radius, _ = target_ball
        # Draw a label for the target ball
        cv2.putText(frame, "Target Ball", (center[0] - 30, center[1] - radius - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw the advanced trajectory
        if cue_ball and cue_line:
            try:
                vx, vy, _, _ = cue_line
                
                # Calculate advanced trajectories using the physics
                trajectories = calculate_advanced_collision(
                    cue_ball, 
                    center,
                    cue_radius,
                    radius,
                    (vx, vy)
                )
                
                if trajectories:
                    if trajectories['will_collide']:
                        # Draw line from cue stick to collision point (white)
                        cv2.line(frame, trajectories['cue_path'][0], 
                                trajectories['collision_point'], (255, 255, 255), 2)
                        
                        # Draw cue ball trajectory after collision (white)
                        cv2.line(frame, trajectories['collision_point'], 
                                trajectories['cue_path'][3], (255, 255, 255), 2)
                        
                        # Draw target ball trajectory (white)
                        cv2.line(frame, trajectories['collision_point'], 
                                trajectories['target_path'][1], (255, 255, 255), 2)
                        
                        # Mark collision point
                        cv2.circle(frame, trajectories['collision_point'], 3, (0, 255, 255), -1)
                    else:
                        # No collision - just draw the straight path
                        cv2.line(frame, trajectories['cue_path'][0], 
                                trajectories['cue_path'][1], (255, 255, 255), 2)
                    
            except Exception as e:
                print(f"Error drawing trajectories: {e}")
                import traceback
                traceback.print_exc()
    
    # Update projection system if provided
    if projection_system is not None and trajectories is not None:
        # Extract lines from trajectories
        lines = extract_lines_from_trajectories(trajectories)
        
        # Update the projection system with these lines
        projection_system.update_lines(lines)
    
    return frame

def main():
    """Main function for the Pool Tracker with Line Projection."""
    print("Starting Pool Tracker with Line Projection System...")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pool tracker with Line Projection')
    parser.add_argument('--video', type=str, default='', help='Path to video file (if not specified, camera will be used)')
    parser.add_argument('--camera', type=int, default=1, help='Camera index (default: 1)')
    parser.add_argument('--output', type=str, default='', help='Path to save output video (optional)')
    parser.add_argument('--projection', action='store_true', help='Enable line projection system')
    parser.add_argument('--use-matplotlib', action='store_true', help='Use Matplotlib for projection (default: OpenCV)')
    args = parser.parse_args()

    # Initialize video capture - either from camera or video file
    if args.video:
        cap = cv2.VideoCapture(args.video)
        print(f"Reading from video file: {args.video}")
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        print(f"Reading from camera index: {args.camera}")

    # Check if the video/camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:  # Sometimes FPS is not available
        fps = 30

    # Initialize video writer if output is specified
    output_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))
        print(f"Saving output to: {args.output}")

    # Define the ArUco dictionary and detector parameters
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Initialize line projection system if enabled
    projection_system = None
    if args.projection:
        shared_data = {'frame_size': (frame_width, frame_height)}
        projection_system = LineProjectionSystem(shared_data)
        projection_system.run(use_matplotlib=args.use_matplotlib)
        print("Line Projection System initialized.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to grab frame.")
                break
            
            # Process the frame with optional projection system
            processed_frame = process_frame(frame, detector, projection_system)
            
            # Display the frame
            cv2.imshow("Pool Tracker", processed_frame)
            
            # Write the frame to output file if specified
            if output_writer is not None:
                output_writer.write(processed_frame)
            
            # Exit on 'q' key press or if ESC is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            
            # If reading from a video and 'p' is pressed, pause/resume
            if args.video and key == ord('p'):
                while True:
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 == ord('p'):  # Resume on 'p'
                        break
                    if key2 == ord('q') or key2 == 27:  # Quit on 'q' or ESC
                        return
                        
            # Add a small delay to make sure the projection system can keep up
            time.sleep(0.01)
    finally:
        # Clean up
        cap.release()
        if output_writer is not None:
            output_writer.release()
        
        # Stop the projection system if it was created
        if projection_system is not None:
            projection_system.stop()
            
        cv2.destroyAllWindows()
        print("Pool Tracker stopped.")

if __name__ == "__main__":
    main()