# ATEBall

This repository contains the code for the 2024-2025 USC Makers project, ATE Ball. The goal of this project is to detect the positions of each ball on a pool table and generate trajectory lines that replicate Apple's GamePigeon 8-Ball. The code runs on our modified pool table, bringing the digital GamePigeon 8-Ball experience into real-life gameplay.

## How To Use

### With projection enabled
python projection.py --projection

### Use a video file for testing
python projection.py --video videos/shot.mp4

- add video testing

- cue ball moves fast and often, make sure its position is rechecked often

- crop to aruco tags on window pane

- automatically full screen projection on board

- make sure projected line disappears when no longer picking up a line

- make sure straight lines are projected as well as three lines


