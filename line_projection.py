import cv2
import numpy as np
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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