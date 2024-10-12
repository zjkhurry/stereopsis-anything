# Stereo Anything

**Stereo Anything** can convert 2D content on the screen in real-time into stereoscopic images (spatial videos) that are theoretically compatible with various AR/VR glasses, such as Rayneo Air 1s/2s, X1, X2, Nreal Air, etc. The project is specifically optimized for macOS and is designed to work best on Mac systems.
![stereo image](img/1.jpeg)
![stereo image](img/2.gif)

## Features

- Real-time conversion of 2D content on the screen to stereoscopic images.
- Optimized for macOS.
- Utilizes Pyobjc to call Screen Capture Kit for direct access to screen shots from the image buffer of macOS.
- A specially optimized CoreML [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) model that supports tensor input/output.
- Renders graphics directly on the GPU using OpenGL.
- Achieves 33 FPS with approximately 100ms latency on M3 Max, suitable for smooth video playback.
- Achieves around 10 FPS on M2, slightly laggy but still usable :satisfied:.
![delay](img/delay.jpeg)

## Technology Stack

- **Python**: For development and application building.
- **Pyobjc**: To interact with macOS system-level functionalities.
- **CoreML**: For deep learning model inference.
- **OpenGL**: For efficient graphics rendering.
- **Screen Capture Kit**: For capturing screen content.
- **Pytorch**: For high performance MPS acceleration.

## Installation Guide

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/zjkhurry/stereo-anything.git
   cd stereo-anything
   ```

2. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

## Usage Guide

Run the main program:
```bash
python3 run.py
```
The program will crop the center area of the main display with a ratio of 16:9 (the area for a full screen 1080p video), anddisplay a window with the stereoscopic video (3840 x 1080). Set your AR glasses to 3D mode, extend the screen, and then drag the stereoscopic video window to the extended screen for full-screen display. Press 'q' to quit the app and ',' and '.' to modify the depth of the stereo image. 

## Notes

- Ensure your macOS version > macOS 13.0 and Mac device supports OpenGL 2.1.
- Due to high performance requirements, it is recommended to use supported hardware devices (such as M3 Max or M2 Max chips) for the best experience.
