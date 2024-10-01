import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to convert pixels to inches
def pixels_to_inches(pixels, distance_to_camera, image_width, focal_length):
    return (pixels * distance_to_camera) / focal_length

# Function to determine size recommendation using a point system
def get_size_recommendation(shoulder_width, chest_width, sleeve_length, gender):
    #TTaaga Man Panjabi(Slim Fit)
    size_chart_male = {
        'S': {'shoulder_width': (14, 16), 'chest_width': (32, 40), 'sleeve_length': (22, 24.5)},
        'M': {'shoulder_width': (16, 17), 'chest_width': (40, 42), 'sleeve_length': (24.5, 24.75)},
        'L': {'shoulder_width': (17, 17.5), 'chest_width': (42, 44), 'sleeve_length': (24.75, 25)},
        'XL': {'shoulder_width': (17.5, 18), 'chest_width': (44, 46), 'sleeve_length': (25, 25.5)},
        'XXL': {'shoulder_width': (18, 18.5), 'chest_width': (46, 48), 'sleeve_length': (25.5, 26.5)},
    }

    size_chart_female = {
        'S': {'shoulder_width': (13, 15), 'chest_width': (30, 32), 'sleeve_length': (21, 23)},
        'M': {'shoulder_width': (15, 17), 'chest_width': (32, 36), 'sleeve_length': (23, 25)},
        'L': {'shoulder_width': (17, 19), 'chest_width': (36, 40), 'sleeve_length': (25, 27)},
        'XL': {'shoulder_width': (19, 21), 'chest_width': (40, 44), 'sleeve_length': (27, 29)},
        'XXL': {'shoulder_width': (21, 23), 'chest_width': (44, 48), 'sleeve_length': (29, 31)},
    }

    size_chart = size_chart_male if gender == 'male' else size_chart_female

    points = {size: 0 for size in size_chart.keys()}

    for size, measurements in size_chart.items():
        if measurements['shoulder_width'][0] <= shoulder_width <= measurements['shoulder_width'][1]:
            points[size] += 1
        if measurements['chest_width'][0] <= chest_width <= measurements['chest_width'][1]:
            points[size] += 1
        if measurements['sleeve_length'][0] <= sleeve_length <= measurements['sleeve_length'][1]:
            points[size] += 1

    max_points = max(points.values())
    recommended_sizes = [size for size, point in points.items() if point == max_points]

    if recommended_sizes:
        return recommended_sizes[0]
    return 'Size not found'

# Known parameters(Change as the environment setup)
distance_to_camera = 63  # in inches
focal_length = 1080  # in pixels (approximated for 1080p camera)

#System Calibration Factors
sleeve_factor = 0.8989


# Initialize webcam
cap = cv2.VideoCapture("http://admin:admin@172.29.35.211/cgi-bin/guest/Video.cgi?media=MJPEG&profile=1&token=csrf1234&rnd=0.9784416519577905/video")

# Set up full screen window
cv2.namedWindow('Webcam', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Webcam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

gender = 'male'  # Default gender

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image and find the pose
        results = pose.process(image)

        # Draw the pose annotation on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract relevant coordinates
            landmarks = results.pose_landmarks.landmark
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            print(right_hip)

            # Convert normalized coordinates to pixel values
            h, w, _ = image.shape
            left_shoulder = [int(left_shoulder[0] * w), int(left_shoulder[1] * h)]
            right_shoulder = [int(right_shoulder[0] * w), int(right_shoulder[1] * h)]
            left_elbow = [int(left_elbow[0] * w), int(left_elbow[1] * h)]
            right_elbow = [int(right_elbow[0] * w), int(right_elbow[1] * h)]
            left_wrist = [int(left_wrist[0] * w), int(left_wrist[1] * h)]
            right_wrist = [int(right_wrist[0] * w), int(right_wrist[1] * h)]
            left_hip = [int(left_hip[0] * w), int(left_hip[1] * h)]
            right_hip = [int(right_hip[0] * w), int(right_hip[1] * h)]
            print("######", right_hip)

            # Calculate distances in pixels
            shoulder_distance_px = calculate_distance(left_shoulder, right_shoulder)
            chest_height_px = calculate_distance(
                [(left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2],
                [(left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2]
            )
            chest_width_px = 1.60 * chest_height_px  # Approximating chest width based on height
            sleeve_length_px = calculate_distance(left_shoulder, left_wrist)

            # Convert distances from pixels to inches
            shoulder_distance_inch = pixels_to_inches(shoulder_distance_px, distance_to_camera, w, focal_length)
            chest_distance_inch = pixels_to_inches(chest_width_px, distance_to_camera, w, focal_length)
            sleeve_length_inch = pixels_to_inches(sleeve_length_px, distance_to_camera, w, focal_length)            
            
            #Calibration
            #actual_sleeve_length_inch = 22 #1st time user's
            #measured_sleeve_length_inch = sleeve_length_inch
            #calibration_factor = actual_sleeve_length_inch / measured_sleeve_length_inch
            #print("Sleeve Factor will be: ",calibration_factor)
            
            sleeve_length_inch = sleeve_length_inch * sleeve_factor
            
            
            # Get size recommendation
            size_recommendation = get_size_recommendation(shoulder_distance_inch, chest_distance_inch, sleeve_length_inch, gender)

            # Display the size recommendation and measurements
            cv2.putText(image, f'Size Recommendation: {size_recommendation}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Shoulder Width: {shoulder_distance_inch:.2f} inches', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Chest Width: {chest_distance_inch:.2f} inches', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Sleeve Length: {sleeve_length_inch:.2f} inches', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Webcam', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()