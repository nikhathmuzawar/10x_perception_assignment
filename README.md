# Cuboid Rotation Analyzer

This project estimates the **orientation** and **visible area** of a rotating cuboid using **depth images** recorded in a ROS2 bag file (`.db3` format).  
It detects multiple planar surfaces (faces of the cuboid) using **iterative RANSAC**, computes their area and normal vectors, and identifies the **dominant visible face** at each frame.  
Finally, it estimates the **axis of rotation** from the change in normals over time.  
---
## Requirements

Tested on **Python 3.10+** and **ROS2 Humble** data bags.

### Install dependencies:
```bash
pip install numpy opencv-python rosbags scipy
