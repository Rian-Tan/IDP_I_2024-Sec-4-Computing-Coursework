from imutils import face_utils
import dlib
import cv2
import numpy as np

# Vamos inicializar um detector de faces (HOG) para então
# let's go code an faces detector(HOG) and after detect the 
# landmarks on this detected face

# p = our pre-treined model directory, on my case, it's on the same script's diretory.
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

def find_valid_port():
    start_port = -100
    end_port = 100

    for port in range(start_port, end_port + 1):
        cap = cv2.VideoCapture(port)
        
        if cap.isOpened():
            print(f"Successfully opened port {port}")
            return cap
        else:
            print(f"Failed to open port {port}")

    print("Unable to find a valid port.")
    return None
    
# Call the function to find a valid port
cap = find_valid_port()

# Load sunglasses image
sunglasses_img = cv2.imread('sunglasses.png', -1)

def overlay_sunglasses(face_image, sunglasses_image, landmarks):
    # Extract relevant points for the eyes from landmarks
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]

    # Calculate the width of the sunglasses
    sunglasses_width = int(np.linalg.norm(right_eye[0] - left_eye[3]))

    # Resize sunglasses image to match the calculated width
    sunglasses_resized = cv2.resize(sunglasses_image, (sunglasses_width*2, int(sunglasses_width / sunglasses_image.shape[1] * sunglasses_image.shape[0])*2))

    # Calculate the position to overlay sunglasses on the face
    x_offset = left_eye[0][0]
    y_offset = int((left_eye[0][1] + right_eye[3][1]) / 2) - int(sunglasses_resized.shape[0] / 2)

    # Calculate the angle of rotation
    angle = np.arctan2(right_eye[3][1] - left_eye[0][1], right_eye[3][0] - left_eye[0][0]) * 180 / np.pi

    # Rotate sunglasses image in the opposite direction
    rotation_matrix = cv2.getRotationMatrix2D((sunglasses_resized.shape[1] // 2, sunglasses_resized.shape[0] // 2), -angle, 1)
    rotated_sunglasses = cv2.warpAffine(sunglasses_resized, rotation_matrix, (sunglasses_resized.shape[1], sunglasses_resized.shape[0]))






    # Overlay rotated sunglasses on the face
    for c in range(0, 3):
        face_image[y_offset:y_offset + rotated_sunglasses.shape[0], x_offset:x_offset + rotated_sunglasses.shape[1], c] = \
            rotated_sunglasses[:, :, c] * (rotated_sunglasses[:, :, 3] / 255.0) + \
            face_image[y_offset:y_offset + rotated_sunglasses.shape[0], x_offset:x_offset + rotated_sunglasses.shape[1], c] * (1.0 - rotated_sunglasses[:, :, 3] / 255.0)

    return face_image

while True:
    # Getting out image by webcam 
    _, image = cap.read()

    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Get faces into webcam's image
    rects = detector(gray, 0)
    
    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Make the prediction and transform it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Overlay sunglasses on the detected face
        image = overlay_sunglasses(image, sunglasses_img, shape)

        # Draw on our image, all the found coordinate points (x,y)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    # Show the image
    cv2.imshow("Output", image)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
