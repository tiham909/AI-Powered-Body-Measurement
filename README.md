# Virtual Tailor: AI-Powered Clothing Size Recommendation System

## Overview
Virtual Tailor is a computer vision application that uses MediaPipe and OpenCV to recommend clothing sizes based on body measurements. The system captures real-time video, detects body landmarks, calculates key measurements (shoulder width, chest width, and sleeve length), and recommends the best-fitting size according to predefined size charts.

## Features
- Real-time body landmark detection using MediaPipe Pose
- Measurement calculation of:
  - Shoulder width
  - Chest width
  - Sleeve length
- Gender-specific size recommendations (male/female)
- Calibration system for improved accuracy
- Full-screen display mode for better visibility

## Technical Specifications
- **Programming Language**: Python
- **Main Libraries**:
  - OpenCV (cv2) for video capture and image processing
  - MediaPipe for pose estimation
  - NumPy for mathematical operations
- **Camera Requirements**: Works with standard webcams or IP cameras

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/virtual-tailor.git
   cd virtual-tailor
   ```

2. Install the required packages:
   ```bash
   pip install opencv-python mediapipe numpy
   ```

## Usage
1. Run the application:
   ```bash
   python virtual_tailor.py
   ```

2. For IP camera setup:
   - Modify the video capture URL in the code:
     ```python
     cap = cv2.VideoCapture("http://admin:admin@your_camera_ip/...")
     ```

3. System calibration:
   - Adjust the following parameters in the code for your environment:
     ```python
     distance_to_camera = 63  # Distance from camera to subject in inches
     focal_length = 1080     # Camera focal length in pixels (approximated)
     sleeve_factor = 0.8989  # Calibration factor for sleeve length
     ```

4. Gender selection:
   - Change the default gender by modifying:
     ```python
     gender = 'male'  # or 'female'
     ```

## Size Charts
The system uses predefined size charts for both male and female clothing:

### Male Size Chart (Slim Fit)
| Size | Shoulder Width (in) | Chest Width (in) | Sleeve Length (in) |
|------|---------------------|------------------|--------------------|
| S    | 14-16               | 32-40            | 22-24.5            |
| M    | 16-17               | 40-42            | 24.5-24.75         |
| L    | 17-17.5             | 42-44            | 24.75-25           |
| XL   | 17.5-18             | 44-46            | 25-25.5            |
| XXL  | 18-18.5             | 46-48            | 25.5-26.5          |

### Female Size Chart
| Size | Shoulder Width (in) | Chest Width (in) | Sleeve Length (in) |
|------|---------------------|------------------|--------------------|
| S    | 13-15               | 30-32            | 21-23              |
| M    | 15-17               | 32-36            | 23-25              |
| L    | 17-19               | 36-40            | 25-27              |
| XL   | 19-21               | 40-44            | 27-29              |
| XXL  | 21-23               | 44-48            | 29-31              |

## Calibration Process
For accurate measurements:
1. Measure the actual sleeve length of a reference user
2. Run the system to get the measured sleeve length
3. Calculate the calibration factor:
   ```python
   calibration_factor = actual_sleeve_length / measured_sleeve_length
   ```
4. Update the sleeve_factor in the code

## Troubleshooting
- If landmarks aren't detected:
  - Ensure proper lighting
  - User should stand straight with arms slightly away from body
  - Check camera feed quality
- If measurements seem inaccurate:
  - Recalibrate the system
  - Verify distance_to_camera and focal_length values

## Future Enhancements
- Integration with e-commerce platforms
- Support for more clothing types
- 3D body modeling
- Mobile app version

## Acknowledgments
- MediaPipe by Google for the pose estimation model
- OpenCV community for computer vision tools
