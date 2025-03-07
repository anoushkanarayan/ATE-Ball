import cv2
import numpy as np
import argparse

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
    # Detect ArUco markers and get boundary coordinates
    frame, aruco_mask, bounds = detect_aruco_markers(frame, detector)
    
    # Detect cue ball
    cue_ball, cue_radius = detect_white_ball(frame, aruco_mask)
    
    # Draw the cue ball
    if cue_ball:
        cv2.circle(frame, cue_ball, cue_radius, (0, 255, 0), 2)
        cv2.putText(frame, "Cue Ball", (cue_ball[0] - 30, cue_ball[1] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
        # Detect cue orientation
        cue_line, is_pointing_at_ball = detect_cue_orientation(frame, cue_ball, aruco_mask)
        
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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    markerCorners, markerIds, _ = detector.detectMarkers(gray)
    
    # Create a mask to exclude ArUco marker regions
    aruco_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    
    if markerIds is not None:
        all_corners = []
        for corners, marker_id in zip(markerCorners, markerIds):
            corners = corners.reshape((4, 2)).astype(int)
            all_corners.extend(corners)
            
            # Draw the ArUco markers on the frame
            cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, f"ID: {marker_id[0]}", (corners[0][0], corners[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Create a filled polygon mask for the ArUco marker
            cv2.fillPoly(aruco_mask, [corners], 0)
        
        # Get bounding box around all detected markers
        if all_corners:
            x_min = min([c[0] for c in all_corners])
            y_min = min([c[1] for c in all_corners])
            x_max = max([c[0] for c in all_corners])
            y_max = max([c[1] for c in all_corners])
            bounds = (x_min, y_min, x_max, y_max)
        else:
            bounds = (0, 0, frame.shape[1], frame.shape[0])
    else:
        bounds = (0, 0, frame.shape[1], frame.shape[0])
    
    return frame, aruco_mask, bounds

def detect_white_ball(frame, aruco_mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # HSV range for white
    lower_white, upper_white = np.array([0, 0, 200]), np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Apply the ArUco mask to exclude those regions
    mask = cv2.bitwise_and(mask, aruco_mask)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Filter contours by circularity and size
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            # Avoid division by zero
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.7 and area > 500:  # Adjust threshold as needed
                    valid_contours.append(contour)
        
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            if radius > 10:
                return (int(x), int(y)), int(radius)
    return None, None

def detect_cue_orientation(frame, cue_ball, aruco_mask):
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


def find_intersecting_ball(frame, cue_line, aruco_mask):
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