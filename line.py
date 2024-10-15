# This code will print a plane black linne, this is to test that we can project lines

import matplotlib.pyplot as plt

# Create a figure and axis
fig, ax = plt.subplots()

# Create a black line across the middle of the figure
ax.plot([0, 1], [0.5, 0.5], color='black', linewidth=5)  # Horizontal black line

# Remove axes for clean projection
ax.set_axis_off()

# Set background color to white
fig.patch.set_facecolor('white')

# Set the figure to fullscreen mode for the projector
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

# Show the plot
plt.show()
