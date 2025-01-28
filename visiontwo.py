import cv2
import numpy as np

# Initialize both cameras
cap0 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap1 = cv2.VideoCapture(2, cv2.CAP_DSHOW)

# Define the ArUco dictionary and detector parameters
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Function to detect ArUco markers and draw bounding boxes
def detect_aruco_markers(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(gray)

    if markerIds is not None:
        for corners, marker_id in zip(markerCorners, markerIds):
            corners = corners.reshape((4, 2)).astype(int)  # Ensure corners are integers
            top_left, top_right, bottom_right, bottom_left = corners

            # Draw lines between the corners to form a rectangle
            cv2.line(frame, tuple(top_left), tuple(top_right), (0, 255, 0), 2)
            cv2.line(frame, tuple(top_right), tuple(bottom_right), (0, 255, 0), 2)
            cv2.line(frame, tuple(bottom_right), tuple(bottom_left), (0, 255, 0), 2)
            cv2.line(frame, tuple(bottom_left), tuple(top_left), (0, 255, 0), 2)

            # Put the marker ID near the top-left corner
            text_position = (top_left[0], top_left[1] - 10)  # Ensure text is above the marker
            cv2.putText(frame, f"ID: {marker_id[0]}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, markerCorners, markerIds

# Function to detect the white pool ball
def detect_white_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for white color in HSV
    lower_white = np.array([0, 0, 200])  # Adjusted for white hue
    upper_white = np.array([180, 30, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Find contours for the detected white regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if contours:
        # Find the largest contour (assuming it's the pool ball)
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

        # Only consider it a ball if the radius is above a threshold
        if radius > 10:
            center = (int(x), int(y))
            cv2.circle(frame, center, int(radius), (0, 255, 0), 2)

    return frame, center

# Function to detect the blue cue stick and its orientation
def detect_cue_orientation_and_draw(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for blue color in HSV
    lower_blue = np.array([100, 50, 150])  # Decrease Saturation and Increase Value
    upper_blue = np.array([140, 200, 255])  # Optionally refine Saturation and keep Value maxed out

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours for the blue stick
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Use the largest contour to approximate the stick
        largest_contour = max(contours, key=cv2.contourArea)
        [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)

        # Calculate the endpoints of the stick's line
        start_point = (int(x - vx * 500), int(y - vy * 500))  # Extend in the negative direction
        end_point = (int(x + vx * 500), int(y + vy * 500))    # Extend in the positive direction

        # Draw the stick's line on the frame
        cv2.line(frame, start_point, end_point, (255, 0, 0), 3)

        # Calculate the angle of orientation in degrees
        angle = np.arctan2(vy, vx) * 180 / np.pi
        return int(angle), frame

    return None, frame

# Create a window for the projector display with a dynamic line
def create_projector_window(angle, fixed_start=(260, 270), line_length=700, window_width=1280, window_height=800):
    projector_background = np.full((window_height, window_width, 3), (50, 50, 50), dtype=np.uint8)

    if angle is not None:
        # Calculate the end point of the line based on the angle
        angle_rad = np.deg2rad(angle)
        end_x = int(fixed_start[0] + line_length * np.cos(angle_rad))
        end_y = int(fixed_start[1] + line_length * np.sin(angle_rad))

        # Draw the line
        cv2.line(projector_background, fixed_start, (end_x, end_y), (255, 255, 255), 5)

    return projector_background

# Main loop
while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if not ret0 or not ret1:
        print("Failed to grab frames from both cameras.")
        break

    # Camera 0: Detect ArUco markers and white ball
    frame0, corners0, ids0 = detect_aruco_markers(frame0)
    frame0, center0 = detect_white_ball(frame0)

    # Camera 1: Detect blue cue stick
    cue_angle, frame1 = detect_cue_orientation_and_draw(frame1)

    # Combine the two feeds side by side for visualization
    combined_frame = np.hstack((frame1, frame0))

    # Create and display the projector window
    projector_window = create_projector_window(cue_angle)
    cv2.imshow("Projector Display", projector_window)

    # Move the window to the projector screen (adjust position as needed)
    cv2.moveWindow("Projector Display", 1470, 0)  # Assumes projector is to the right of the main screen

    # Display the combined frame
    cv2.imshow('Combined View', combined_frame)

    # Break the loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap0.release()
cap1.release()
cv2.destroyAllWindows()

