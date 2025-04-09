# ATEBall

This repository contains the code for the 2024-2025 USC Makers project, ATE Ball. The goal of this project is to detect the positions of each ball on a pool table and generate trajectory lines that replicate Apple's GamePigeon 8-Ball. The code runs on our modified pool table, bringing the digital GamePigeon 8-Ball experience into real-life gameplay.

## How To Use

### With projection enabled
python projection.py --projection

### Save the output to a video file
python projection.py --output output.avi

### Use a video file for testing
python projection.py --video videos/shot.mp4

- cue ball moves fast and often, make sure its position is rechecked often

- reflect projection window 180 degrees

- automatically full screen projection on board

- make sure projected line disappears when no longer picking up a line

- projection needs to tweak more, projection from camera feed doesnt update as often as projection view

- make sure straight lines are projected as well as three lines


