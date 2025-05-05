import cv2
import numpy as np
import argparse
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

LOCKED_BOUNDS = [None]
LOCKED_TABLE_MASK = [None]
SHARED_CUE_BALL = {
    'positions': [None, None],  # One position per camera
    'radii': [None, None],      # One radius per camera
    'confidences': [0.0, 0.0],  # One confidence value per camera
    'timestamp': time.time()
}
SHARED_MARKERS = []

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
    
    # OFFSET
    def extend_line(start_point, end_point, extension=60):
        """Extend a line by the specified number of pixels."""
        # Calculate direction vector
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        
        # Normalize
        magnitude = (dx**2 + dy**2)**0.5
        if magnitude > 0:
            dx = dx / magnitude
            dy = dy / magnitude
        
        # Extend end point
        extended_end = (
            int(end_point[0] + dx * extension),
            int(end_point[1] + dy * extension)
        )
        
        return extended_end
    
    if trajectories.get('will_collide', False):
        # Original collision point (for prong calculations)
        orig_collision = transform_point(trajectories['collision_point'])
        
        # Cue stick to collision point
        start = transform_point(trajectories['cue_path'][0])
        extended_collision = extend_line(start, orig_collision)
        
        # Add the extended stem line
        lines.append((start[0]+0, start[1]+20, extended_collision[0]+0, extended_collision[1]+20))
        
        # Get original trajectories for the prongs
        cue_after = transform_point(trajectories['cue_path'][3])
        target_after = transform_point(trajectories['target_path'][1])
        
        # Calculate the prong vector directions exactly as they were
        cue_vector = (cue_after[0] - orig_collision[0], cue_after[1] - orig_collision[1])
        target_vector = (target_after[0] - orig_collision[0], target_after[1] - orig_collision[1])
        
        # Apply these same vectors but starting from the extended collision point
        new_cue_end = (extended_collision[0] + cue_vector[0], extended_collision[1] + cue_vector[1])
        new_target_end = (extended_collision[0] + target_vector[0], extended_collision[1] + target_vector[1])
        
        # Add the two prong lines with the original vector math
        lines.append((extended_collision[0]+0, extended_collision[1]+20, new_cue_end[0]+0, new_cue_end[1]+20))
        lines.append((extended_collision[0]+0, extended_collision[1]+20, new_target_end[0]+0, new_target_end[1]+20))
    else:
        # No collision - just the straight path
        start = transform_point(trajectories['cue_path'][0])
        end = transform_point(trajectories['cue_path'][1])
        # Extend this line too since it's the stem line in this case
        extended_end = extend_line(start, end)
        lines.append((start[0], start[1], extended_end[0], extended_end[1]))
    
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
    
    if len(LOCKED_BOUNDS) <= camera_index:
        # Extend the list to accommodate the camera index
        LOCKED_BOUNDS.extend([None] * (camera_index + 1 - len(LOCKED_BOUNDS)))
    
    if len(LOCKED_TABLE_MASK) <= camera_index:
        # Extend the list to accommodate the camera index
        LOCKED_TABLE_MASK.extend([None] * (camera_index + 1 - len(LOCKED_TABLE_MASK)))
    
    # If we have locked bounds and mask for this camera, use them
    if camera_index < len(LOCKED_BOUNDS) and LOCKED_BOUNDS[camera_index] is not None and \
       camera_index < len(LOCKED_TABLE_MASK) and LOCKED_TABLE_MASK[camera_index] is not None:
        # Still detect markers for coordination purposes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        markerCorners, markerIds, _ = detector.detectMarkers(gray)
        
        # Create new aruco mask - WHITE (255) background with BLACK (0) markers
        aruco_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        
        marker_data = []
        if markerIds is not None:
            # Draw the detected markers on the frame
            frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            
            # Fill markers with BLACK (0) in the mask
            for corners in markerCorners:
                corners = corners.reshape((4, 2)).astype(int)
                cv2.fillPoly(aruco_mask, [corners], 0)
            
            # Store marker data
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
                aruco_mask,
                LOCKED_BOUNDS[camera_index], 
                LOCKED_TABLE_MASK[camera_index],
                marker_data)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ArUco markers
    markerCorners, markerIds, _ = detector.detectMarkers(gray)
    
    # Create aruco mask - WHITE (255) background with BLACK (0) markers
    aruco_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    
    # Default bounds - whole frame
    bounds = (0, 0, frame.shape[1], frame.shape[0])
    
    # Default table mask - whole frame
    table_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    
    # Store marker data for coordination
    marker_data = []
    
    if markerIds is not None:
        # Draw the detected markers
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        
        # Fill markers with BLACK (0) in the mask
        for corners in markerCorners:
            corners = corners.reshape((4, 2)).astype(int)
            cv2.fillPoly(aruco_mask, [corners], 0)
        
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
                
                # Lock in detected values if needed
                if camera_index < len(LOCKED_BOUNDS):
                    LOCKED_BOUNDS[camera_index] = bounds
                else:
                    # Extend and set
                    LOCKED_BOUNDS.extend([None] * (camera_index + 1 - len(LOCKED_BOUNDS)))
                    LOCKED_BOUNDS[camera_index] = bounds
                
                if camera_index < len(LOCKED_TABLE_MASK):
                    LOCKED_TABLE_MASK[camera_index] = table_mask
                else:
                    # Extend and set
                    LOCKED_TABLE_MASK.extend([None] * (camera_index + 1 - len(LOCKED_TABLE_MASK)))
                    LOCKED_TABLE_MASK[camera_index] = table_mask
            else:
                print(f"Warning: Invalid table bounds after applying margin. Using full frame.")
    
    return frame, aruco_mask, bounds, table_mask, marker_data

def detect_white_ball(frame, aruco_mask, table_mask, bounds, camera_index=0):
    """
    Detect the white cue ball in the frame with improved robustness.
    
    Args:
        frame: Input frame
        aruco_mask: Mask for ArUco markers
        table_mask: Mask for table area
        bounds: Table boundaries
        camera_index: Index of the camera (0 or 1)
        
    Returns:
        Tuple of ((center_x, center_y), radius) and a confidence value
    """
    global SHARED_CUE_BALL
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Get average brightness to adapt detection parameters
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    
    # Adapt HSV range based on lighting conditions
    if avg_brightness < 120:  # Darker environment
        lower_white = np.array([0, 0, 170]) 
        upper_white = np.array([180, 50, 255])
    else:  # Brighter environment
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255]) 
    
    # Create mask for white regions
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Apply table mask
    mask = cv2.bitwise_and(mask, table_mask)
    
    # Make sure aruco_mask is properly formatted (255 for non-marker areas, 0 for markers)
    if np.mean(aruco_mask[aruco_mask > 0]) < 128:  # If the non-zero values are mostly dark
        aruco_mask_for_exclusion = cv2.bitwise_not(aruco_mask)
    else:
        aruco_mask_for_exclusion = aruco_mask.copy()
    
    # Ensure the mask is binary (only 0 and 255 values)
    _, aruco_mask_for_exclusion = cv2.threshold(aruco_mask_for_exclusion, 128, 255, cv2.THRESH_BINARY)
    
    # Apply aruco mask to exclude the markers
    mask = cv2.bitwise_and(mask, aruco_mask_for_exclusion)
    
    # Apply morphological operations to improve detection
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
    
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
                    valid_contours.append((contour, circularity, area))
        
        if valid_contours:
            # Get the contour with highest circularity and reasonable size
            best_contour = max(valid_contours, key=lambda x: x[1] * (x[2] / 1000.0 if x[2] < 2000 else 2.0))
            contour, circularity, area = best_contour
            
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            
            if radius > 10 and radius < 30:  # Reasonable radius range for cue ball
                # Calculate confidence based on circularity and size
                size_factor = min(1.0, area / 700.0)  # Normalize by expected area
                confidence = circularity * size_factor
                
                center_point = (int(x), int(y))
                radius_int = int(radius)
                
                # Check if center is within bounds and not on an ArUco tag
                within_bounds = (
                    bounds[0] <= x <= bounds[2] and
                    bounds[1] <= y <= bounds[3]
                )
                
                if within_bounds:
                    # Double check not on ArUco marker
                    if (y < aruco_mask_for_exclusion.shape[0] and 
                        x < aruco_mask_for_exclusion.shape[1] and 
                        aruco_mask_for_exclusion[int(y), int(x)] > 0):
                        
                        # Update shared cue ball info
                        update_shared_cue_ball(center_point, radius_int, confidence, camera_index)
                        return (center_point, radius_int), confidence
    
    # If no ball is detected with the standard approach, try with more relaxed parameters
    # This is especially helpful when the cue stick is near the ball
    lower_white_relaxed = np.array([0, 0, 160])  # Lower threshold for value
    upper_white_relaxed = np.array([180, 60, 255])  # Higher threshold for saturation
    
    relaxed_mask = cv2.inRange(hsv, lower_white_relaxed, upper_white_relaxed)
    relaxed_mask = cv2.bitwise_and(relaxed_mask, table_mask)
    relaxed_mask = cv2.bitwise_and(relaxed_mask, aruco_mask_for_exclusion)
    
    # Apply stronger morphological operations
    kernel = np.ones((5, 5), np.uint8)
    relaxed_mask = cv2.morphologyEx(relaxed_mask, cv2.MORPH_OPEN, kernel)
    relaxed_mask = cv2.morphologyEx(relaxed_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the relaxed mask
    contours, _ = cv2.findContours(relaxed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Filter by circularity and size
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200:  # Even more relaxed size threshold
                continue
                
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.6:  # More relaxed circularity
                    valid_contours.append((contour, circularity, area))
        
        if valid_contours:
            # Get the most circular contour
            best_contour = max(valid_contours, key=lambda x: x[1])
            contour, circularity, area = best_contour
            
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            
            if radius > 8 and radius < 35:  # More relaxed radius range
                center_point = (int(x), int(y))
                radius_int = int(radius)
                
                # Calculate a lower confidence for relaxed detection
                confidence = circularity * 0.8  # Reduce confidence for relaxed parameters
                
                # Check if within bounds and not on ArUco marker
                within_bounds = (
                    bounds[0] <= x <= bounds[2] and
                    bounds[1] <= y <= bounds[3]
                )
                
                if within_bounds and y < aruco_mask_for_exclusion.shape[0] and x < aruco_mask_for_exclusion.shape[1]:
                    if aruco_mask_for_exclusion[int(y), int(x)] > 0:
                        # Update shared cue ball info with lower confidence
                        update_shared_cue_ball(center_point, radius_int, confidence, camera_index)
                        return (center_point, radius_int), confidence
    
    return None, 0.0

def update_shared_cue_ball(center, radius, confidence, camera_index):
    """
    Update the shared cue ball information across cameras.
    
    Args:
        center: Center point of the cue ball
        radius: Radius of the cue ball
        confidence: Detection confidence (0-1)
        camera_index: Index of the camera (0 or 1)
    """
    global SHARED_CUE_BALL
    
    # Make sure we have a valid center point
    if center is None or not isinstance(center, tuple) or len(center) != 2:
        return
    
    # Initialize the data structure if it doesn't exist
    if SHARED_CUE_BALL is None or not isinstance(SHARED_CUE_BALL, dict):
        SHARED_CUE_BALL = {
            'positions': [None, None],
            'radii': [None, None],
            'confidences': [0.0, 0.0],
            'timestamp': time.time()
        }
    
    # Make sure arrays exist and have correct length
    for key in ['positions', 'radii', 'confidences']:
        if key not in SHARED_CUE_BALL or not isinstance(SHARED_CUE_BALL[key], list):
            SHARED_CUE_BALL[key] = [None, None] if key != 'confidences' else [0.0, 0.0]
        
        # Extend arrays if needed
        while len(SHARED_CUE_BALL[key]) <= camera_index:
            if key == 'confidences':
                SHARED_CUE_BALL[key].append(0.0)
            else:
                SHARED_CUE_BALL[key].append(None)
    
    # Update this camera's information
    SHARED_CUE_BALL['positions'][camera_index] = center
    SHARED_CUE_BALL['radii'][camera_index] = radius
    SHARED_CUE_BALL['confidences'][camera_index] = confidence
    SHARED_CUE_BALL['timestamp'] = time.time()
    
    # Clear old information (older than 0.5 seconds)
    current_time = time.time()
    if current_time - SHARED_CUE_BALL.get('timestamp', 0) > 0.5:
        # Reset confidences for old detections
        for i in range(len(SHARED_CUE_BALL['confidences'])):
            if i != camera_index:
                SHARED_CUE_BALL['confidences'][i] = 0.0

def get_best_cue_ball():
    """
    Get the best cue ball detection across cameras.
    
    Returns:
        Tuple of (center, radius, camera_index) or (None, None, None)
    """
    global SHARED_CUE_BALL
    
    # Check if SHARED_CUE_BALL is initialized properly
    if SHARED_CUE_BALL is None or not isinstance(SHARED_CUE_BALL, dict):
        return None, None, None
    
    # Make sure required keys exist
    required_keys = ['positions', 'radii', 'confidences']
    if not all(key in SHARED_CUE_BALL for key in required_keys):
        return None, None, None
    
    # Find camera with highest confidence
    max_confidence = 0.0
    best_index = -1
    
    # Get the length of the confidences list to avoid index errors
    conf_length = len(SHARED_CUE_BALL['confidences'])
    
    # Make sure we're only iterating up to the available length
    for i in range(min(conf_length, 2)):  # Limit to max 2 cameras
        if i < conf_length:  # Double-check index is valid
            confidence = SHARED_CUE_BALL['confidences'][i]
            if confidence > max_confidence:
                max_confidence = confidence
                best_index = i
    
    # Make sure best_index is valid and arrays have enough elements
    if (best_index >= 0 and max_confidence > 0 and 
        best_index < len(SHARED_CUE_BALL['positions']) and 
        best_index < len(SHARED_CUE_BALL['radii'])):
        
        # Make sure the position is not None
        if SHARED_CUE_BALL['positions'][best_index] is not None:
            return (
                SHARED_CUE_BALL['positions'][best_index],
                SHARED_CUE_BALL['radii'][best_index],
                best_index
            )
    
    return None, None, None

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

    lower_red1 = np.array([0, 100, 100])    # Lower red range (beginning of spectrum)
    upper_red1 = np.array([10, 255, 255])   # Upper red range 

    lower_red2 = np.array([160, 100, 100])  # Lower red range (end of spectrum)
    upper_red2 = np.array([180, 255, 255])  # Upper red range

    # Create two masks and combine them
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply aruco mask
    mask = cv2.bitwise_and(mask, aruco_mask)
    
    # ADDED: Apply table mask if provided
    if table_mask is not None:
        mask = cv2.bitwise_and(mask, table_mask)
    
    # ADDED: Apply morphological operations to improve detection
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # ADDED: Create a debug image to visualize the mask
    debug_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    if contours:
        # Find contours with sufficient area and elongated shape
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Skip very small contours
                continue
                
            # Calculate aspect ratio to find elongated shapes
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Cue sticks are elongated objects
            if aspect_ratio > 4.0 or aspect_ratio < 0.25:
                valid_contours.append(contour)
        
        # If we have valid contours, use the largest one
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
        else:
            # Fall back to the largest contour if no valid ones found
            largest_contour = max(contours, key=cv2.contourArea)
        
        # Draw the contour on debug image
        cv2.drawContours(debug_mask, [largest_contour], 0, (0, 255, 0), 2)
        
        # Fit a line to the contour
        [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Extract scalar values
        vx_scalar = float(vx[0])
        vy_scalar = float(vy[0])
        x_scalar = float(x[0])
        y_scalar = float(y[0])
        
        # Ensure the cue direction points away from the ball
        # (Invert direction if needed)
        cue_x, cue_y = cue_ball
        vec_to_ball_x, vec_to_ball_y = cue_x - x_scalar, cue_y - y_scalar
        dot_product = vec_to_ball_x * vx_scalar + vec_to_ball_y * vy_scalar
        
        # If dot product is negative, flip the direction
        if dot_product < 0:
            vx_scalar, vy_scalar = -vx_scalar, -vy_scalar
        
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
        
        # Draw line representing detected cue stick (both forward and backward from point)
        lefty = int((-x_scalar * (vy_scalar/vx_scalar) + y_scalar) if vx_scalar != 0 else 0)
        righty = int(((frame.shape[1]-x_scalar) * (vy_scalar/vx_scalar) + y_scalar) if vx_scalar != 0 else frame.shape[0])
        
        # Calculate endpoints for a line segment of reasonable length
        line_length = 400  # pixels
        
        # Forward line (in direction of cue)
        x1 = int(x_scalar - vx_scalar * line_length)
        y1 = int(y_scalar - vy_scalar * line_length)
        x2 = int(x_scalar + vx_scalar * line_length)
        y2 = int(y_scalar + vy_scalar * line_length)
        
        # Draw the detected cue stick line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        
        # Add a label for the cue stick line
        label_x = x2 + int(vx_scalar * 20)
        label_y = y2 + int(vy_scalar * 20)
        cv2.putText(frame, "Cue Stick", (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
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

def process_frame(frame, detector, camera_index=0, projection_system=None):
    """
    Process a single frame, detect balls and trajectories.
    
    Args:
        frame: Input frame
        detector: ArUco marker detector
        camera_index: Camera index (0 or 1)
        projection_system: Optional LineProjectionSystem
    
    Returns:
        Tuple of (processed frame, marker data, detected balls, trajectories)
    """
    # Flip the frame 180 degrees
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Make a copy for visualization
    original_frame = frame.copy()
    
    # Detect ArUco markers
    frame, aruco_mask, bounds, table_mask, marker_data = detect_aruco_markers(
        frame, detector, camera_index
    )
    
    # Detect cue ball in this frame
    cue_ball_this_frame, confidence = detect_white_ball(
        frame, aruco_mask, table_mask, bounds, camera_index
    )
    
    # Get the best cue ball detection across cameras
    best_cue_ball, best_radius, best_camera = get_best_cue_ball()
    
    # Use the detected cue ball in this frame if available, otherwise use the shared one
    if cue_ball_this_frame and isinstance(cue_ball_this_frame, tuple) and len(cue_ball_this_frame) == 2:
        cue_ball = cue_ball_this_frame[0]
        cue_radius = cue_ball_this_frame[1]
    elif best_cue_ball:
        cue_ball = best_cue_ball
        cue_radius = best_radius if best_radius else 15  # default radius if not detected
    else:
        cue_ball = None
        cue_radius = 15  # default radius if not detected
    
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
    if cue_ball and isinstance(cue_ball, tuple) and len(cue_ball) == 2:
        # If this is not the camera that detected the cue ball, draw with a different color
        cue_color = (0, 255, 0) if camera_index == best_camera else (0, 165, 255)
        
        cv2.circle(frame, cue_ball, cue_radius, cue_color, 2)
        cv2.putText(frame, "Cue Ball", (cue_ball[0] - 30, cue_ball[1] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cue_color, 2)
        
        # Only detect cue orientation if this is the camera that best sees the cue ball
        if camera_index == best_camera or best_camera is None:
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
        # Extract trajectory lines
        lines = extract_lines_from_trajectories(trajectories)
        
        # Update projection system
        projection_system.update_lines(lines)
    
    # Draw camera index and common marker info on frame
    cv2.putText(frame, f"Camera {camera_index+1}", (20, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
    
    # Add info about shared markers
    if SHARED_MARKERS:
        cv2.putText(frame, f"Common Markers: {len(SHARED_MARKERS)}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
    
    # Return processed frame and detection data
    return frame, marker_data, all_detected_balls, trajectories

def main():
    """Main function for the Pool Table Tracker with Side-by-Side View."""
    global SHARED_CUE_BALL, SHARED_MARKERS, LOCKED_BOUNDS, LOCKED_TABLE_MASK
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pool Table Tracker with Side-by-Side View')
    parser.add_argument('--video', type=str, default='', help='Path to video file')
    parser.add_argument('--cam1', type=int, default=0, help='First camera index (default: 0)')
    parser.add_argument('--cam2', type=int, default=2, help='Second camera index (default: 2)')
    parser.add_argument('--output', type=str, default='', help='Path to save output video')
    parser.add_argument('--projection', action='store_true', help='Enable line projection')
    parser.add_argument('--use-matplotlib', action='store_true', help='Use Matplotlib for projection')
    args = parser.parse_args()
    
    if args.video:
        print("Starting Pool Table Tracker using video file...")
    else:
        print("Starting Dual Camera Pool Table Tracker with Side-by-Side View...")
    
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
        
        # Combined dimensions for side-by-side display
        combined_width = frame_width1 + frame_width2
        combined_height = max(frame_height1, frame_height2)
    
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
        try:
            shared_data = {
                'frame_size': (combined_width, combined_height),
                'table_dimensions': (table_width, table_height)
            }
            projection_system = LineProjectionSystem(shared_data)
            projection_system.run(use_matplotlib=args.use_matplotlib)
            print("Line Projection System initialized.")
        except Exception as e:
            print(f"Error initializing projection system: {e}")
            projection_system = None
    
    # Create windows
    if args.video:
        window_title = "Pool Tracker - Video"
    else:
        window_title = "Dual Camera Pool Tracker - Side by Side"
    
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, combined_width, combined_height)
    
    try:
        while True:
            # Read first frame/camera
            ret1, frame1 = cap1.read()
            if not ret1:
                print("End of video or failed to grab frames.")
                break
            
            # Process first frame
            processed_frame1, markers1, balls1, trajectories1 = process_frame(
                frame1, detector, 0, projection_system
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
                    frame2, detector, 1, projection_system
                )
                
                
                # Create side-by-side view
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
                
                # Draw information about common markers and shared cue ball
                cv2.putText(combined_frame, f"Common Markers: {len(SHARED_MARKERS)}", 
                           (10, max_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Get current best cue ball
                best_cue_ball, best_radius, best_camera = get_best_cue_ball()
                if best_cue_ball:
                    camera_text = f"Camera {best_camera + 1}"
                    cv2.putText(combined_frame, f"Cue Ball: {camera_text}", 
                               (10, max_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
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
        
        try:
            if projection_system is not None:
                projection_system.stop()
        except Exception as e:
            print(f"Error stopping projection system: {e}")
        
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error destroying windows: {e}")
            
        print("Pool Tracker stopped.")