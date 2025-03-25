# ATEBall

This repository contains the code for the 2024-2025 USC Makers project, ATE Ball. The goal of this project is to detect the positions of each ball on a pool table and generate trajectory lines that replicate Apple's GamePigeon 8-Ball. The code runs on our modified pool table, bringing the digital GamePigeon 8-Ball experience into real-life gameplay.

How to test on the video:
python [filename].py --projection --video [video].mp4      // filename is single-camera.py till camera issue 

Options:

--projection: Enable the line projection system 
--video PATH: Specify a video file to process instead of using a camera
--camera INDEX: Specify which camera to use (default: 1)
--output PATH: Save the processed video to a file
--use-matplotlib: Use Matplotlib for projection display (default uses OpenCV)

