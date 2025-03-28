import cv2
import numpy as np
import argparse
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Global variables to track state
LOCKED_BOUNDS = [None, None]  # One for each camera
LOCKED_TABLE_MASK = [None, None]  # One for each camera
STITCHING_HOMOGRAPHY = None  # Homography matrix for stitching camera views
SHARED_MARKERS = []  # ArUco markers visible in both camera feeds

class LineProjectionSystem:
    def __init__(self, shared_data=None):
        """
        Initialize the line projection system.
        
        Args:
            shared_data: A dictionary to share data with the main application
        """
        # Create matplotlib figure for projection
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.ax.set_facecolor('black')  # Set axis background to black
        self.fig.patch.set_facecolor('black')  # Set figure background to black
        
        # Get the table dimensions from shared data if available
        self.table_width = 1280
        self.table_height = 640
        if shared_data and 'table_dimensions' in shared_data:
            self.table_width, self.table_height = shared_data['table_dimensions']
            
        self.ax.set_xlim(0, self.table_width)
        self.ax.set_ylim(self.table_height, 0)  # Invert Y-axis to match OpenCV coordinates
        self.ax.set_axis_off()  # Hide axis
        
        # Draw table boundaries
        self.draw_table_boundaries()
        
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
        
        # Flag to indicate if we have lines to display
        self.has_lines = False
        
        # OpenCV window name for projection view
        self.window_name = "Projection View"
        
    def draw_table_boundaries(self):
        """Draw the boundaries of the pool table."""
        # Draw a rectangle representing the pool table
        self.table_rect = plt.Rectangle((0, 0), self.table_width, self.table_height, 
                               fill=False, edgecolor='forestgreen', linewidth=2)
        self.ax.add_patch(self.table_rect)
        
    def update_lines(self, lines):
        """Update detected lines with thread lock."""
        with self.lock:
            self.detected_lines = lines.copy() if lines else []
            # Set flag based on whether we have lines
            self.has_lines = len(self.detected_lines) > 0
            
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
            has_lines = self.has_lines
            
        # Only draw lines if we have lines to display
        if has_lines:
            # Draw each line
            for x1, y1, x2, y2 in current_lines:
                line, = self.ax.plot([x1, x2], [y1, y2], color='white', linewidth=3)
                self.lines.append(line)
            
        # Update shared data if needed
        if 'frame_ready' in self.shared_data:
            self.shared_data['frame_ready'] = True
            
        return self.lines
    
    def create_projection_frame(self):
        """Create a projection frame with detected lines."""
        # Create a black frame with the same dimensions as the table
        projection_frame = np.zeros((self.table_height, self.table_width, 3), dtype=np.uint8)
        
        # Draw a green rectangle for the table boundary
        cv2.rectangle(projection_frame, (0, 0), (self.table_width, self.table_height), 
                      (0, 100, 0), 2)
        
        # Get current detected lines with thread lock
        with self.lock:
            current_lines = self.detected_lines.copy()
            has_lines = self.has_lines
        
        # Only draw lines if we have lines to display
        if has_lines:
            # Draw each line in white
            for x1, y1, x2, y2 in current_lines:
                cv2.line(projection_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                         (255, 255, 255), 5)
            
        return projection_frame
        
    def run_opencv_display(self):
        """Run OpenCV-based display for projection."""
        # Set window properties
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.table_width, self.table_height)
        
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
def extract_lines_from_trajectories(trajectories, transform_matrix=None):
    """
    Extract line segments from trajectory data.
    
    Args:
        trajectories: Dictionary containing trajectory data
        transform_matrix: Optional homography matrix to transform points
        
    Returns:
        List of line segments as (x1, y1, x2, y2) tuples
    """
    lines = []
    
    if not trajectories:
        return lines
    
    def transform_point(point):
        """Transform a point using the homography matrix if provided."""
        if transform_matrix is not None:
            # Convert to homogeneous coordinates
            p = np.array([[point[0], point[1], 1]], dtype=np.float32).T
            # Apply transformation
            p_transformed = np.dot(transform_matrix, p)
            # Convert back from homogeneous coordinates
            return (int(p_transformed[0]/p_transformed[2]), 
                    int(p_transformed[1]/p_transformed[2]))
        return point
    
    if trajectories.get('will_collide', False):
        # Cue stick to collision point
        start = transform_point(trajectories['cue_path'][0])
        end = transform_point(trajectories['collision_point'])
        lines.append((start[0], start[1], end[0], end[1]))
        
        # Cue ball after collision
        start = transform_point(trajectories['collision_point']) 
        end = transform_point(trajectories['cue_path'][3])
        lines.append((start[0], start[1], end[0], end[1]))
        
        # Target ball trajectory
        start = transform_point(trajectories['collision_point'])
        end = transform_point(trajectories['target_path'][1])
        lines.append((start[0], start[1], end[0], end[1]))
    else:
        # No collision - just the straight path
        start = transform_point(trajectories['cue_path'][0])
        end = transform_point(trajectories['cue_path'][1])
        lines.append((start[0], start[1], end[0], end[1]))
    
    return lines

def calculate_advanced_collision(cue_pos, target_pos, cue_radius, target_radius, stick_direction):
    """
    Calculate advanced collision trajectories between cue ball and target ball
    using physics equations.
    
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

def detect_aruco_markers(frame, detector, camera_index=0):
    """
    Detect ArUco markers in the frame.
    
    Args:
        frame: Input frame
        detector: ArUco marker detector
        camera_index: Index of the camera (0 or 1)
        
    Returns:
        Tuple of (annotated frame, ArUco mask, table bounds, table mask, marker data)
    """
    global LOCKED_BOUNDS, LOCKED_TABLE_MASK, SHARED_MARKERS
    
    # If we have locked bounds and mask for this camera, use them
    if LOCKED_BOUNDS[camera_index] is not None and LOCKED_TABLE_MASK[camera_index] is not None:
        # Still detect markers for stitching purposes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        markerCorners, markerIds, _ = detector.detectMarkers(gray)
        
        marker_data = []
        if markerIds is not None:
            # Draw the detected markers
            frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            
            # Store marker data for stitching
            for i, (corners, marker_id) in enumerate(zip(markerCorners, markerIds)):
                corners = corners.reshape((4, 2)).astype(int)
                # Calculate center of marker
                center_x = int(np.mean([c[0] for c in corners]))
                center_y = int(np.mean([c[1] for c in corners]))
                
                marker_data.append({
                    'id': int(marker_id[0]),
                    'corners': corners,
                    'center': (center_x, center_y)
                })
        
        return (frame, 
                np.ones(frame.shape[:2], dtype=np.uint8) * 255, 
                LOCKED_BOUNDS[camera_index], 
                LOCKED_TABLE_MASK[camera_index],
                marker_data)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ArUco markers
    markerCorners, markerIds, _ = detector.detectMarkers(gray)
    
    # Default mask and bounds
    aruco_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    
    # Default bounds - whole frame
    bounds = (0, 0, frame.shape[1], frame.shape[0])
    
    # Default table mask - whole frame
    table_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    
    # Store marker data for stitching
    marker_data = []
    
    if markerIds is not None:
        # Draw the detected markers
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        
        # Process each marker
        all_corners = []
        for i, (corners, marker_id) in enumerate(zip(markerCorners, markerIds)):
            corners = corners.reshape((4, 2)).astype(int)
            all_corners.extend(corners)
            
            # Calculate center of marker
            center_x = int(np.mean([c[0] for c in corners]))
            center_y = int(np.mean([c[1] for c in corners]))
            
            # Add marker data
            marker_data.append({
                'id': int(marker_id[0]),
                'corners': corners,
                'center': (center_x, center_y)
            })
            
            # Draw marker ID
            cv2.putText(frame, f"ID: {marker_id[0]}", (center_x - 20, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if all_corners and len(all_corners) >= 4:
            # For a precise table boundary, use the exact corners of the markers
            # Find the outermost corner points to make a tight bounding rectangle
            x_coords = [c[0] for c in all_corners]
            y_coords = [c[1] for c in all_corners]
            
            # Get min and max values to form the bounds
            x_min = max(0, min(x_coords))
            y_min = max(0, min(y_coords))
            x_max = min(frame.shape[1], max(x_coords))
            y_max = min(frame.shape[0], max(y_coords))
            
            # Apply a negative margin of 25 pixels to make the box smaller
            margin = -25  # Negative margin to shrink the boundary
            x_min = max(0, x_min - margin)  # Adding negative margin increases the value
            y_min = max(0, y_min - margin)  # Adding negative margin increases the value
            x_max = min(frame.shape[1], x_max + margin)  # Adding negative margin decreases the value
            y_max = min(frame.shape[0], y_max + margin)  # Adding negative margin decreases the value
            
            # Ensure we have a valid box after applying the margin
            if x_max > x_min and y_max > y_min:
                bounds = (x_min, y_min, x_max, y_max)
                
                # Create a precise table mask using the bounds
                table_points = np.array([[x_min, y_min], [x_max, y_min], 
                                        [x_max, y_max], [x_min, y_max]])
                
                table_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(table_mask, [table_points], 255)
                
                # Draw table boundary
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                
                # Lock in detected values
                LOCKED_BOUNDS[camera_index] = bounds
                LOCKED_TABLE_MASK[camera_index] = table_mask
            else:
                print(f"Warning: Invalid table bounds after applying margin. Using full frame.")
    
    return frame, aruco_mask, bounds, table_mask, marker_data

def compute_stitching_homography(markers1, markers2):
    """
    Compute homography matrix to stitch two camera views.
    
    Args:
        markers1: Marker data from first camera
        markers2: Marker data from second camera
        
    Returns:
        Homography matrix to transform points from second camera to first
    """
    global SHARED_MARKERS
    
    # Find common markers between the two cameras
    common_markers = []
    
    # Create dictionaries for fast lookup
    markers1_dict = {m['id']: m for m in markers1}
    markers2_dict = {m['id']: m for m in markers2}
    
    # Find common marker IDs
    common_ids = set(markers1_dict.keys()).intersection(set(markers2_dict.keys()))
    
    if len(common_ids) < 4:
        print(f"Warning: Only {len(common_ids)} common markers found. At least 4 are recommended for accurate stitching.")
    
    # Get marker centers
    if common_ids:
        src_points = np.array([markers2_dict[id]['center'] for id in common_ids], dtype=np.float32)
        dst_points = np.array([markers1_dict[id]['center'] for id in common_ids], dtype=np.float32)
        
        # Store shared markers for reference
        SHARED_MARKERS = list(common_ids)
        
        # Compute homography if we have enough points
        if len(common_ids) >= 4:
            H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
            return H
    
    # If not enough common markers, return identity matrix
    return np.eye(3, dtype=np.float32)

def detect_white_ball(frame, aruco_mask, table_mask, bounds):
    """Detect the white cue ball in the frame."""
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # HSV range for white ball
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])
    
    # Create mask for white regions
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Apply table mask
    mask = cv2.bitwise_and(mask, table_mask)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Filter by circularity and size
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 300:  # Skip small contours
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.7:  # Circle-like shape
                    valid_contours.append(contour)
        
        if valid_contours:
            # Get the largest valid contour
            largest_contour = max(valid_contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            
            if radius > 10:  # Minimum radius threshold
                return (int(x), int(y)), int(radius)
    
    return None, None

def detect_colored_balls(frame, aruco_mask, table_mask, bounds, cue_ball=None):
    """
    Detect colored balls in the frame.
    
    Args:
        frame: Input frame
        aruco_mask: Mask for ArUco markers
        table_mask: Mask for table area
        bounds: Table boundaries (x_min, y_min, x_max, y_max)
        cue_ball: Position and radius of cue ball, to exclude from detection
        
    Returns:
        List of (center, radius, color) tuples for each detected ball
    """
    # Convert to HSV for color analysis
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Convert to grayscale for circle detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply table mask to grayscale image
    masked_gray = cv2.bitwise_and(gray, gray, mask=table_mask)
    
    # Apply Gaussian blur
    masked_gray = cv2.GaussianBlur(masked_gray, (5, 5), 0)
    
    # Detect circles using Hough transform
    circles = cv2.HoughCircles(
        masked_gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=25,
        minRadius=10,
        maxRadius=30
    )
    
    colored_balls = []
    
    if circles is not None:
        # Convert to integer coordinates
        circles = np.uint16(np.around(circles))
        
        for circle in circles[0, :]:
            x, y, radius = circle
            center = (int(x), int(y))
            
            # Skip if this is potentially the cue ball
            if cue_ball and distance(center, cue_ball) < radius * 1.5:
                continue
                
            # Skip if not within bounds
            if (x - radius < bounds[0] or
                y - radius < bounds[1] or
                x + radius > bounds[2] or
                y + radius > bounds[3]):
                continue
                
            # Skip if center is outside table mask
            if y < table_mask.shape[0] and x < table_mask.shape[1] and table_mask[y, x] == 0:
                continue
                
            # Create a mask for this ball
            ball_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(ball_mask, center, radius, 255, -1)
            
            # Apply table mask
            ball_mask = cv2.bitwise_and(ball_mask, table_mask)
            
            # Skip if masked area is too small
            if np.sum(ball_mask) < np.pi * radius * radius * 0.7:
                continue
            
            # Get average color
            masked_hsv = cv2.bitwise_and(hsv, hsv, mask=ball_mask)
            if np.sum(ball_mask) > 0:
                hsv_values = masked_hsv[ball_mask > 0]
                mean_color = np.mean(hsv_values, axis=0).astype(int)
                
                # Skip if it's white (likely the cue ball)
                h, s, v = mean_color
                if s < 40 and v > 180:
                    continue
                
                # Add to colored balls list
                colored_balls.append((center, radius, mean_color))
    
    return colored_balls

def get_color_name(hsv_color):
    """Convert HSV color values to a human-readable name."""
    h, s, v = hsv_color
    
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
    """
    Detect the orientation of the cue stick.
    
    Args:
        frame: Input frame
        cue_ball: Position of cue ball
        aruco_mask: Mask for ArUco markers
        table_mask: Mask for table area
        
    Returns:
        Tuple of (cue parameters, is_pointing_at_ball)
    """
    if cue_ball is None:
        return None, False
    
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # HSV range for cue stick (light colors)
    lower_silver = np.array([0, 0, 180])
    upper_silver = np.array([180, 30, 255])
    
    # Create mask for potential cue stick
    mask = cv2.inRange(hsv, lower_silver, upper_silver)
    
    # Apply aruco mask
    mask = cv2.bitwise_and(mask, aruco_mask)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour (likely the cue stick)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit a line to the contour
        [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Extract scalar values
        vx_scalar = float(vx[0])
        vy_scalar = float(vy[0])
        x_scalar = float(x[0])
        y_scalar = float(y[0])
        
        # Ensure the cue direction points away from the ball
        # (Invert direction if needed)
        vx_scalar, vy_scalar = -vx_scalar, -vy_scalar
        
        # Get cue ball position
        cue_x, cue_y = cue_ball
        
        # Calculate perpendicular distance from cue line to ball center
        perp_x, perp_y = -vy_scalar, vx_scalar  # Perpendicular to direction
        vec_to_ball_x, vec_to_ball_y = cue_x - x_scalar, cue_y - y_scalar
        
        # Distance from line to point
        distance_to_line = abs(vec_to_ball_x * perp_x + vec_to_ball_y * perp_y) / np.sqrt(perp_x**2 + perp_y**2)
        
        # Check if ball is ahead of the cue, not behind
        dot_product = vec_to_ball_x * vx_scalar + vec_to_ball_y * vy_scalar
        is_ahead = dot_product > 0
        
        # Maximum allowed distance (can be adjusted based on ball radius)
        max_distance = 20  # pixels
        
        is_pointing_at_ball = is_ahead and distance_to_line < max_distance
        
        # Return line parameters and pointing status
        return (vx_scalar, vy_scalar, cue_x, cue_y), is_pointing_at_ball
    
    return None, False

def find_target_ball(cue_ball, cue_line, all_balls):
    """
    Find the ball that the cue is pointing at.
    
    Args:
        cue_ball: Position of cue ball
        cue_line: Parameters of cue line
        all_balls: List of all detected balls
        
    Returns:
        Ball information (center, radius, color) or None
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
        
        # Calculate perpendicular distance from ball to cue line
        if vx == 0:  # Vertical line
            perp_distance = abs(ball_x - x0)
        else:
            perp_distance = abs(vy*(ball_x - x0) - vx*(ball_y - y0)) / np.sqrt(vx*vx + vy*vy)
        
        # Distance from cue ball to this ball
        ball_distance = distance(cue_ball, center)
        
        # Check if ball is in front of cue ball
        ball_vector = [ball_x - cue_x, ball_y - cue_y]
        direction_vector = [vx, vy]
        dot_product = ball_vector[0]*direction_vector[0] + ball_vector[1]*direction_vector[1]
        
        # Ball must be in front of cue and close to the line
        if dot_product > 0 and perp_distance < radius + 5:
            if ball_distance < min_ball_distance:
                min_ball_distance = ball_distance
                closest_ball = ball_info
    
    return closest_ball

def transform_point(point, homography=None):
    """
    Transform a point using a homography matrix.
    
    Args:
        point: (x, y) coordinates
        homography: Homography transformation matrix
        
    Returns:
        Transformed point
    """
    if homography is None:
        return point
        
    # Convert to homogeneous coordinates
    p = np.array([[point[0], point[1], 1]], dtype=np.float32).T
    
    # Apply transformation
    p_transformed = np.dot(homography, p)
    
    # Convert back from homogeneous coordinates
    return (int(p_transformed[0]/p_transformed[2]), 
            int(p_transformed[1]/p_transformed[2]))

def process_frame(frame, detector, camera_index=0, homography=None, projection_system=None):
    """
    Process a single frame, detect balls and trajectories.
    
    Args:
        frame: Input frame
        detector: ArUco marker detector
        camera_index: Camera index (0 or 1)
        homography: Homography matrix for stitching
        projection_system: Optional LineProjectionSystem
    
    Returns:
        Tuple of (processed frame, marker data, detected balls, trajectories)
    """
    # Make a copy for visualization
    original_frame = frame.copy()
    
    # Detect ArUco markers
    frame, aruco_mask, bounds, table_mask, marker_data = detect_aruco_markers(
        frame, detector, camera_index
    )
    
    # Detect cue ball
    cue_ball, cue_radius = detect_white_ball(frame, aruco_mask, table_mask, bounds)
    
    # Detect colored balls
    colored_balls = detect_colored_balls(frame, aruco_mask, table_mask, bounds, cue_ball)
    
    # Store all detected balls
    all_detected_balls = []
    
    # Process colored balls
    for ball_center, ball_radius, ball_color in colored_balls:
        # Draw circle around ball
        cv2.circle(frame, ball_center, ball_radius, (0, 0, 255), 2)
        
        # Get color name
        color_name = get_color_name(ball_color)
        
        # Store ball information
        all_detected_balls.append((ball_center, ball_radius, color_name))
    
    # Initialize trajectory variables
    target_ball = None
    trajectories = None
    
    # Draw cue ball and detect cue orientation
    if cue_ball:
        cv2.circle(frame, cue_ball, cue_radius, (0, 255, 0), 2)
        cv2.putText(frame, "Cue Ball", (cue_ball[0] - 30, cue_ball[1] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Detect cue orientation
        cue_line, is_pointing_at_ball = detect_cue_orientation(
            frame, cue_ball, aruco_mask, table_mask
        )
        
        # Only process if cue is pointing at the ball
        if cue_line and is_pointing_at_ball:
            vx, vy, x0, y0 = cue_line
            
            # Find target ball
            target_ball = find_target_ball(cue_ball, cue_line, all_detected_balls)
            
            # If no target ball, draw regular cue line
            if not target_ball:
                # Calculate end point
                end_point = (int(x0 + vx * 400), int(y0 + vy * 400))
                start_point = (int(x0), int(y0))
                
                # Draw line
                cv2.line(frame, start_point, end_point, (255, 255, 255), 2)
    
    # If target ball found, draw trajectory
    if target_ball:
        center, radius, _ = target_ball
        
        # Label target ball
        cv2.putText(frame, "Target Ball", (center[0] - 30, center[1] - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Calculate and draw trajectory
        if cue_ball and cue_line:
            try:
                vx, vy, _, _ = cue_line
                
                # Calculate collision trajectory
                trajectories = calculate_advanced_collision(
                    cue_ball,
                    center,
                    cue_radius,
                    radius,
                    (vx, vy)
                )
                
                if trajectories:
                    if trajectories['will_collide']:
                        # Draw trajectory lines
                        cv2.line(frame, trajectories['cue_path'][0],
                               trajectories['collision_point'], (255, 255, 255), 2)
                        
                        cv2.line(frame, trajectories['collision_point'],
                               trajectories['cue_path'][3], (255, 255, 255), 2)
                        
                        cv2.line(frame, trajectories['collision_point'],
                               trajectories['target_path'][1], (255, 255, 255), 2)
                        
                        # Mark collision point
                        cv2.circle(frame, trajectories['collision_point'], 3, (0, 255, 255), -1)
                    else:
                        # No collision - straight path
                        cv2.line(frame, trajectories['cue_path'][0],
                               trajectories['cue_path'][1], (255, 255, 255), 2)
            except Exception as e:
                print(f"Error drawing trajectories: {e}")
                import traceback
                traceback.print_exc()
    
    # Update projection system if provided
    if projection_system is not None and trajectories is not None:
        # Extract trajectory lines with appropriate transformation
        lines = extract_lines_from_trajectories(trajectories, homography if camera_index == 1 else None)
        
        # Update projection system
        projection_system.update_lines(lines)
    
    # Draw camera index on frame
    cv2.putText(frame, f"Camera {camera_index+1}", (20, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
    
    # Return processed frame and detection data
    return frame, marker_data, all_detected_balls, trajectories

def create_stitched_view(frame1, frame2, homography):
    """
    Create a stitched view of two camera frames.
    
    Args:
        frame1: First camera frame
        frame2: Second camera frame
        homography: Homography matrix to transform frame2 to frame1's perspective
        
    Returns:
        Stitched frame
    """
    # Get dimensions
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    
    # Warp the second frame to align with the first
    warped_frame2 = cv2.warpPerspective(frame2, homography, (w1, h1))
    
    # Create a mask to blend the overlapping regions
    mask = np.zeros((h1, w1), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.array([[0, 0], [w1//2, 0], [w1//2, h1], [0, h1]]), 255)
    
    # Combine the frames
    stitched = np.copy(warped_frame2)
    stitched[mask == 255] = frame1[mask == 255]
    
    # Draw a line at the seam
    cv2.line(stitched, (w1//2, 0), (w1//2, h1), (0, 255, 255), 1)
    
    return stitched

def main():
    """Main function for the Pool Table Tracker with Projection."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pool Table Tracker with Projection')
    parser.add_argument('--video', type=str, default='', help='Path to video file')
    parser.add_argument('--cam1', type=int, default=0, help='First camera index (default: 0)')
    parser.add_argument('--cam2', type=int, default=2, help='Second camera index (default: 2)')
    parser.add_argument('--output', type=str, default='', help='Path to save output video')
    parser.add_argument('--projection', action='store_true', help='Enable line projection')
    parser.add_argument('--use-matplotlib', action='store_true', help='Use Matplotlib for projection')
    parser.add_argument('--no-stitch', action='store_true', help='Disable stitching, show side-by-side')
    args = parser.parse_args()
    
    if args.video:
        print("Starting Pool Table Tracker with Projection using video...")
    else:
        print("Starting Dual Camera Pool Table Stitcher with Projection...")
    


    # Initialize video capture
    if args.video:
        # When using a video file, treat it as a single camera input
        cap1 = cv2.VideoCapture(args.video)
        cap2 = None  # No second camera when using video
        print(f"Reading from video file: {args.video}")
    else:
        # Use two separate cameras
        cap1 = cv2.VideoCapture(args.cam1, cv2.CAP_DSHOW)
        cap2 = cv2.VideoCapture(args.cam2, cv2.CAP_DSHOW)
        print(f"Reading from cameras: {args.cam1} and {args.cam2}")
    
    # Check if camera/video opened successfully
    if not cap1.isOpened():
        print(f"Error: Could not open first camera/video file.")
        return
    
    # Only check the second camera if not using video
    if not args.video and not cap2.isOpened():
        print(f"Error: Could not open second camera (index {args.cam2}).")
        return
    
    # Get video properties
    frame_width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if args.video:
        # For video mode, just use the dimensions of the video
        combined_width = frame_width1
        combined_height = frame_height1
    else:
        # Get second camera properties
        frame_width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Combined dimensions for dual camera setup
        if args.no_stitch:
            # Side-by-side view
            combined_width = frame_width1 + frame_width2
            combined_height = max(frame_height1, frame_height2)
        else:
            # Stitched view - approximately 1.5x the width of one camera
            combined_width = int(frame_width1 * 1.5)
            combined_height = frame_height1
    
    # Estimated pool table dimensions (for projection)
    table_width = combined_width
    table_height = combined_height
    
    # FPS
    fps = cap1.get(cv2.CAP_PROP_FPS)
    if not args.video:
        fps = max(fps, cap2.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30
    
    # Initialize video writer
    output_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_writer = cv2.VideoWriter(
            args.output, fourcc, fps, (combined_width, combined_height)
        )
        print(f"Saving output to: {args.output}")
    
    # Initialize ArUco detector
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    
    # Initialize line projection system
    projection_system = None
    if args.projection:
        shared_data = {
            'frame_size': (combined_width, combined_height),
            'table_dimensions': (table_width, table_height)
        }
        projection_system = LineProjectionSystem(shared_data)
        projection_system.run(use_matplotlib=args.use_matplotlib)
        print("Line Projection System initialized.")
    
    # Create windows
    if args.video:
        window_title = "Pool Tracker - Video"
    else:
        window_title = "Dual Camera Pool Tracker"
    
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, combined_width, combined_height)
    
    # Variable to store homography matrix
    homography = None
    
    try:
        while True:
            # Read first frame/camera
            ret1, frame1 = cap1.read()
            if not ret1:
                print("End of video or failed to grab frames.")
                break
            
            # Process first frame
            processed_frame1, markers1, balls1, trajectories1 = process_frame(
                frame1, detector, 0, None, projection_system
            )
            
            if args.video:
                # In video mode, just use the single processed frame
                combined_frame = processed_frame1
            else:
                # In dual camera mode, read and process second camera
                ret2, frame2 = cap2.read()
                if not ret2:
                    print("Failed to grab frame from second camera.")
                    break
                
                # Process second frame
                processed_frame2, markers2, balls2, trajectories2 = process_frame(
                    frame2, detector, 1, homography, projection_system
                )
                
                # Compute homography if enough markers are detected
                if markers1 and markers2:
                    homography = compute_stitching_homography(markers1, markers2)
                
                # Create output frame
                if args.no_stitch or homography is None:
                    # Side-by-side view
                    max_height = max(frame_height1, frame_height2)
                    
                    # Resize frames to have the same height
                    scale1 = max_height / frame_height1
                    scale2 = max_height / frame_height2
                    
                    new_width1 = int(frame_width1 * scale1)
                    new_width2 = int(frame_width2 * scale2)
                    
                    resized_frame1 = cv2.resize(processed_frame1, (new_width1, max_height))
                    resized_frame2 = cv2.resize(processed_frame2, (new_width2, max_height))
                    
                    # Combine side by side
                    combined_frame = np.zeros((max_height, new_width1 + new_width2, 3), dtype=np.uint8)
                    combined_frame[:, :new_width1] = resized_frame1
                    combined_frame[:, new_width1:] = resized_frame2
                    
                    # Draw dividing line
                    cv2.line(combined_frame, (new_width1, 0), (new_width1, max_height), (0, 255, 255), 2)
                else:
                    # Create stitched view
                    combined_frame = create_stitched_view(processed_frame1, processed_frame2, homography)
            
            # Display frame
            cv2.imshow(window_title, combined_frame)
            
            # Write frame to output
            if output_writer is not None:
                # Resize if needed
                output_frame = cv2.resize(combined_frame, (combined_width, combined_height))
                output_writer.write(output_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            
            # Pause/resume on 'p' if using video file
            if args.video and key == ord('p'):
                while True:
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 == ord('p'):  # Resume
                        break
                    if key2 == ord('q') or key2 == 27:  # Quit
                        return
            
            # Small delay
            time.sleep(0.01)
            
    finally:
        # Clean up
        cap1.release()
        if cap2 is not None:
            cap2.release()
        
        if output_writer is not None:
            output_writer.release()
        
        if projection_system is not None:
            projection_system.stop()
        
        cv2.destroyAllWindows()
        if args.video:
            print("Pool Tracker stopped.")
        else:
            print("Dual Camera Pool Table Stitcher stopped.")

if __name__ == "__main__":
    main()