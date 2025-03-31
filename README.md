# ATEBall

This repository contains the code for the 2024-2025 USC Makers project, ATE Ball. The goal of this project is to detect the positions of each ball on a pool table and generate trajectory lines that replicate Apple's GamePigeon 8-Ball. The code runs on our modified pool table, bringing the digital GamePigeon 8-Ball experience into real-life gameplay.

# Basic usage with two cameras (indices 0 and 2)
python projection.py

# With projection enabled
python projection.py --projection

# Use side-by-side view instead of stitching
python projection.py --no-stitch

# Save the output to a video file
python projection.py --output output.avi

# Use a video file for testing
python projection.py --video videos/shot.mp4

