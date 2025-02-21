import cv2
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Define the ArUco dictionary and detector parameters
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

def detect_aruco_markers(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    markerCorners, markerIds, _ = detector.detectMarkers(gray)
    if markerIds is not None:
        for corners, marker_id in zip(markerCorners, markerIds):
            corners = corners.reshape((4, 2)).astype(int)
            cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, f"ID: {marker_id[0]}", (corners[0][0], corners[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def detect_white_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # HSV range for white
    lower_white, upper_white = np.array([0, 0, 200]), np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        if radius > 10:
            return (int(x), int(y)), int(radius)
    return None, None

def detect_cue_orientation(frame, cue_ball):
    if cue_ball is None:
        return None
        
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # HSV range for blue (cue)
    lower_blue, upper_blue = np.array([100, 50, 150]), np.array([140, 200, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest blue contour (the cue)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit a line to the cue contour
        [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy = -vx, -vy  # Invert direction by 180 degrees

        
        # Get cue ball position
        cue_x, cue_y = cue_ball
        
        # We return the line parameters if the cue is close to the cue ball
        # This is a simple proximity check that can be improved
        return vx, vy, cue_x, cue_y
        
    return None

def find_intersecting_ball(frame, cue_line):
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
    
    # Find contours of the colored balls
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_distance = float('inf')
    closest_ball = None
    
    for contour in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
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

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break
    
    # Process the frame
    frame = detect_aruco_markers(frame)
    cue_ball, cue_radius = detect_white_ball(frame)
    cue_line = detect_cue_orientation(frame, cue_ball)
    
    # Draw the cue ball
    if cue_ball:
        cv2.circle(frame, cue_ball, cue_radius, (0, 255, 0), 2)
        cv2.putText(frame, "Cue Ball", (cue_ball[0] - 30, cue_ball[1] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw the cue line
    if cue_line:
        vx, vy, x0, y0 = cue_line
        # Calculate points for drawing the line (extend it in the direction of the cue)
        # The line starts at the cue ball and extends forward
        start_point = (int(x0), int(y0))
        end_point = (int(x0 + vx * 1000), int(y0 + vy * 1000))
        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
    
    # Find and mark the target ball
    target_ball = find_intersecting_ball(frame, cue_line)
    if target_ball:
        cv2.circle(frame, target_ball, 15, (0, 0, 255), 3)
        cv2.putText(frame, "Target Ball", (target_ball[0] - 40, target_ball[1] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow("Pool Tracker", frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()