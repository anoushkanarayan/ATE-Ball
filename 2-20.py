import cv2
import numpy as np
import argparse

LOCKED_BOUNDS = None
LOCKED_TABLE_MASK = None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Pool tracker with ArUco markers')
    parser.add_argument('--video', type=str, default='', help='Path to video file (if not specified, camera will be used)')
    parser.add_argument('--camera', type=int, default=1, help='Camera index (default: 1)')
    parser.add_argument('--output', type=str, default='', help='Path to save output video (optional)')
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

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to grab frame.")
                break
            
            # Process the frame
            processed_frame = process_frame(frame, detector)
            
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
    finally:
        # Clean up
        cap.release()
        if output_writer is not None:
            output_writer.release()
        cv2.destroyAllWindows()

def clip_line_to_bounds(start, end, bounds):
    """ Clips a line segment to stay within the given bounds. """
    x_min, y_min, x_max, y_max = bounds
    
    # Ensure the line does not extend past the bounding box
    clipped_end = list(end)
    if clipped_end[0] < x_min:
        clipped_end[0] = x_min
    elif clipped_end[0] > x_max:
        clipped_end[0] = x_max
    
    if clipped_end[1] < y_min:
        clipped_end[1] = y_min
    elif clipped_end[1] > y_max:
        clipped_end[1] = y_max
    
    return tuple(start), tuple(clipped_end)

def process_frame(frame, detector):
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
                
                # Determine color (optional)
                ball_mask = np.zeros(original_frame.shape[:2], dtype=np.uint8)
                cv2.circle(ball_mask, center, radius, 255, -1)
                
                hsv = cv2.cvtColor(original_frame, cv2.COLOR_BGR2HSV)
                masked_hsv = cv2.bitwise_and(hsv, hsv, mask=ball_mask)
                if np.sum(ball_mask) > 0:
                    mean_color = np.mean(masked_hsv[ball_mask > 0], axis=0).astype(int)
                    color_name = get_color_name(mean_color)
                    #cv2.putText(frame, color_name, (center[0] - 20, center[1] - 20), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
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
            
            # Display ball color (optional)
            #color_name = get_color_name(ball_color)
            #cv2.putText(frame, color_name, (ball_center[0] - 20, ball_center[1] - 20), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw the cue ball
    if cue_ball:
        cv2.circle(frame, cue_ball, cue_radius, (0, 255, 0), 2)
        cv2.putText(frame, "Cue Ball", (cue_ball[0] - 30, cue_ball[1] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
        # Detect cue orientation
        cue_line, is_pointing_at_ball = detect_cue_orientation(frame, cue_ball, aruco_mask, table_mask)
        
        # Draw the cue line ONLY if it's pointing at the ball
        if cue_line and is_pointing_at_ball:
            vx, vy, x0, y0 = cue_line
            
            # Calculate end point of the line (extend in cue direction)
            end_point = (int(x0 + vx * 400), int(y0 + vy * 400))
            
            # Clip the line to stay within ArUco marker bounds
            #start_point, clipped_end_point = clip_line_to_bounds((int(x0), int(y0)), end_point, bounds)
            start_point = (int(x0), int(y0))
            
            # Draw the clipped line
            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
    
    return frame

def detect_aruco_markers(frame, detector):
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
    
    # For debugging purposes (comment out in production)
    # cv2.imshow("Ball Detection Debug", debug_frame)
    
    return colored_balls

def distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_dominant_color(frame, hsv, mask):
    """Get the dominant HSV color of a masked region."""
    # Apply mask to HSV image
    masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
    
    # Get all non-zero pixels
    nonzero = masked_hsv[np.nonzero(mask)]
    
    if len(nonzero) > 0:
        # Calculate average HSV values
        mean_color = np.mean(nonzero, axis=0).astype(int)
        return mean_color
    
    return np.array([0, 0, 0])

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
        vx, vy = -vx, -vy  # Invert direction by 180 degrees
        
        # Get the cue ball position
        cue_x, cue_y = cue_ball
        
        # Check if the cue is pointing at the ball
        # Calculate distance from cue line to ball center
        # The cue line starts at (x,y) and goes in direction (vx,vy)
        
        # Project vector from cue line point to ball onto the perpendicular to the cue direction
        perp_x, perp_y = -vy, vx  # Perpendicular to (vx, vy)
        vec_to_ball_x, vec_to_ball_y = cue_x - x, cue_y - y
        
        # Distance from line to point
        distance = abs(vec_to_ball_x * perp_x + vec_to_ball_y * perp_y) / np.sqrt(perp_x**2 + perp_y**2)
        
        # Check if ball is ahead of the cue, not behind
        dot_product = vec_to_ball_x * vx + vec_to_ball_y * vy
        is_ahead = dot_product > 0
        
        # Maximum allowed distance (can be adjusted based on ball radius and tolerance)
        max_distance = 20  # pixels
        
        is_pointing_at_ball = is_ahead and distance < max_distance
        
        # Return the line parameters (direction and origin) and whether it's pointing at the ball
        return (vx, vy, cue_x, cue_y), is_pointing_at_ball
        
    return None, False


def find_intersecting_ball(frame, cue_line, aruco_mask, table_mask=None):
    if cue_line is None:
        return None
        
    vx, vy, x0, y0 = cue_line
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a mask for all colored balls (excluding white)
    # You may need to adjust these values based on your pool table and lighting
    lower_color = np.array([0, 50, 50])
    upper_color = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Exclude the white ball from consideration
    lower_white, upper_white = np.array([0, 0, 200]), np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(white_mask))
    
    # Apply the ArUco mask to exclude those regions
    mask = cv2.bitwise_and(mask, aruco_mask)
    
    # Find contours of the colored balls
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_distance = float('inf')
    closest_ball = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:  # Filter out small contours
            continue
            
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        
        # Check for circularity (to ensure we're detecting balls, not arbitrary shapes)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.7:  # Not circular enough
                continue
        
        if radius > 10:  # Filter small contours
            # Calculate the distance from the line to the center of the ball
            # The line equation is: (y - y0) / (x - x0) = vy / vx
            # Reorganizing: vy*(x - x0) - vx*(y - y0) = 0
            
            # Avoid division by zero
            if vx == 0:
                # Vertical line
                dist = abs(x - x0)
            else:
                # Distance from point to line formula
                dist = abs(vy*(x - x0) - vx*(y - y0)) / np.sqrt(vx*vx + vy*vy)
            
            # Check if the ball is in front of the cue ball (not behind)
            # Calculate the dot product of the vector from cue to ball and the cue direction
            ball_vector = [x - x0, y - y0]
            direction_vector = [vx, vy]
            dot_product = ball_vector[0]*direction_vector[0] + ball_vector[1]*direction_vector[1]
            
            # If the ball is in front of the cue and close to the line
            if dot_product > 0 and dist < radius and dist < min_distance:
                min_distance = dist
                closest_ball = (int(x), int(y))
    
    return closest_ball

if __name__ == "__main__":
    main()