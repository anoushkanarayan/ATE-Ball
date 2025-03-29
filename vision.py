import cv2

def open_webcam():
    # Initialize webcam (0 is typically the built-in webcam)
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam was opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Webcam opened successfully!")
    print("Press 'q' to quit.")
    
    # Create window
    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            # If frame is read correctly ret is True
            if not ret:
                print("Error: Can't receive frame. Exiting...")
                break
            
            # Display the resulting frame
            cv2.imshow('Webcam', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # When everything done, release the capture and destroy windows
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released and window closed.")

if __name__ == "__main__":
    open_webcam()