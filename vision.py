import cv2
import numpy as np

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Change to 1 for external camera if needed

    # Load the ArUco dictionary and set up the detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Variables to manage ArUco detection
    frozen_corners = None  # Stores the detected ArUco marker corners
    detected_marker_ids = set()  # Track detected marker IDs
    frozen_hull = None  # Stores the frozen convex hull
    frozen_hull_smushed = None  # Stores the smushed hull

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)

        # If markers are detected, process them
        if marker_ids is not None and frozen_corners is None:
            detected_marker_ids.update(marker_ids.flatten())

            # If all 4 markers are detected, freeze their corners
            if len(detected_marker_ids) >= 4:
                frozen_corners = [corner[0].astype(int) for corner in marker_corners]
                print("All 4 markers detected and frozen.")

        # Use frozen corners if available
        if frozen_corners is not None and frozen_hull is None:
            # Calculate the convex hull and smushed hull once
            all_corners = np.array([pt for corner in frozen_corners for pt in corner], dtype=np.int32)

            # Compute convex hull
            frozen_hull = cv2.convexHull(all_corners)

            # Calculate the center of the hull
            center_x = np.mean(frozen_hull[:, 0, 0])  # Mean of all x-coordinates
            center_y = np.mean(frozen_hull[:, 0, 1])  # Mean of all y-coordinates

            # Compress y-coordinates symmetrically toward the center
            compression_factor = 0.6  # Adjust this factor to reduce height
            frozen_hull_smushed = frozen_hull.copy()
            for point in frozen_hull_smushed:
                original_y = point[0][1]
                point[0][1] = int(center_y + (original_y - center_y) * compression_factor)

            print("Green frame frozen.")

        # Draw the frozen green frame if available
        if frozen_hull_smushed is not None:
            cv2.polylines(frame, [frozen_hull_smushed], isClosed=True, color=(0, 255, 0), thickness=2)

            # Detect the white cue ball
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 200])  # Adjust if needed
            upper_white = np.array([180, 50, 255])
            ball_mask = cv2.inRange(hsv, lower_white, upper_white)
            contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cue_ball_center = None
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
                if radius > 5:  # Ensure it's not a tiny object
                    cue_ball_center = (int(cx), int(cy))
                    cv2.circle(frame, cue_ball_center, int(radius), (0, 255, 0), 2)

            # Detect the blue tip and cue stick
            lower_blue = np.array([100, 150, 50])  # Adjust thresholds for blue
            upper_blue = np.array([140, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if blue_contours:
                largest_blue_contour = max(blue_contours, key=cv2.contourArea)
                blue_center, blue_radius = cv2.minEnclosingCircle(largest_blue_contour)
                blue_center = (int(blue_center[0]), int(blue_center[1]))

                if blue_radius > 5:  # Ensure it's not a tiny object
                    cv2.circle(frame, blue_center, int(blue_radius), (255, 0, 0), 2)

                    # Find the line approximating the cue stick
                    cue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
                    cue_contours, _ = cv2.findContours(cue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if cue_contours:
                        # Use the largest contour to approximate the stick
                        largest_cue_contour = max(cue_contours, key=cv2.contourArea)
                        [vx, vy, x, y] = cv2.fitLine(largest_cue_contour, cv2.DIST_L2, 0, 0.01, 0.01)
                        lefty = int((-x * vy / vx) + y)
                        righty = int(((frame.shape[1] - x) * vy / vx) + y)
                        cv2.line(frame, (frame.shape[1] - 1, righty), (0, lefty), (0, 0, 255), 2)

                        # Check if the blue tip points at the cue ball
                        if cue_ball_center:
                            distance = np.linalg.norm(np.array(blue_center) - np.array(cue_ball_center))
                            if distance <= 50:  # Adjust this threshold
                                print("YES TRUE")
                            else:
                                print("NO FALSE")

        # Display the processed frame
        cv2.imshow("Frame", frame)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
