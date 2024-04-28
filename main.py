# import necessary libraries for facial feature detection and image processing
#ryan (lines 3-17)
from imutils import face_utils
import dlib
import cv2
import numpy as np
import datetime

# define the path to the pre-trained facial landmark predictor model
p = "shape_predictor_68_face_landmarks.dat"

# initialize the face detector and shape predictor using the pre-trained models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# initialize the video capture object, using the default camera (0) as the video source
cap = cv2.VideoCapture(0);

#loading images of facial props from file paths
#ryan (lines 21-22)
round_glasses = cv2.imread('./images/01_round_glasses.png', -1)
rectangular_glasses = cv2.imread('./images/02_rectangular_glasses.png', -1)
# rian (lines 24-37)
pridebubble = cv2.imread('./images/03_pridebubble.png', -1)
stembubble = cv2.imread('./images/04_stembubble.png', -1)
yesbubble = cv2.imread('./images/05_yesbubble.png', -1)
innovationbubble = cv2.imread('./images/06_innovationbubble.png', -1)
staffpride = cv2.imread('./images/07_staffpride.png', -1)
proudofsst = cv2.imread('./images/08_proudofsst.png', -1)
redbluegrey = cv2.imread('./images/09_redbluegrey.png', -1)
celebrating = cv2.imread('./images/10_celebrating.png', -1)
sstinc = cv2.imread('./images/11_sstinc.png', -1)
sst_infineon = cv2.imread('./images/12_SST-infineon.png', -1)
sstsmu = cv2.imread('./images/13_sstsmu.png', -1)
pforssst = cv2.imread('./images/14_pforssst.png', -1)
top_hat = cv2.imread('./images/15_hat.png', -1)
partyhat = cv2.imread('./images/16_partyhat.png',-1)

#error handling for loading image files
#ryan (lines 41-50)
if (
    round_glasses is None or rectangular_glasses is None or
    pridebubble is None or stembubble is None or yesbubble is None or
    innovationbubble is None or staffpride is None or proudofsst is None or
    redbluegrey is None or celebrating is None or sstinc is None or
    sst_infineon is None or sstsmu is None or pforssst is None or
    top_hat is None or partyhat is None
):
    print("Error loading image files. Please check the file paths.")
    exit()


#rian (lines 54-94)
# function to overlay an image on the detected face
def overlay_image(face_image, overlay, landmarks):
    # landmarks of eyes
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]

    # center point calculation
    center_x = (left_eye[0][0] + right_eye[3][0]) // 2
    center_y = (left_eye[0][1] + right_eye[3][1]) // 2

    # Calculate the width of the overlayed image
    overlay_width = int(np.linalg.norm(right_eye[0] - left_eye[3]))

    # resizing
    overlay_resized = cv2.resize(overlay, (overlay_width*18, int(overlay_width / overlay.shape[1] * overlay.shape[0])*14))

    # position to offset the overlay image
    x_offset = center_x - (overlay_resized.shape[1] // 2)
    y_offset = center_y - (overlay_resized.shape[0] // 2) + 50

    # to calculate the angle of rotation
    angle = np.arctan2(right_eye[3][1] - left_eye[0][1], right_eye[3][0] - left_eye[0][0]) * 180 / np.pi

    # to rotate the overlay
    rotation_matrix = cv2.getRotationMatrix2D((overlay_resized.shape[1] // 2, overlay_resized.shape[0] // 2), -angle, 1)
    rotated_overlay = cv2.warpAffine(overlay_resized, rotation_matrix, (overlay_resized.shape[1], overlay_resized.shape[0]))

    # calculate the ROI (region of interest) to put the overlay on
    roi = face_image[y_offset:y_offset + rotated_overlay.shape[0], x_offset:x_offset + rotated_overlay.shape[1]]

    # Ensure that the shapes of the ROI and rotated overlay match
    if roi.shape[0] != rotated_overlay.shape[0] or roi.shape[1] != rotated_overlay.shape[1]:
        #print("Shapes of ROI and rotated sunglasses do not match.") #error got annoying so i removed it ez
        return face_image

    #combine the overlay image with the face image
    for c in range(0, 3):
        roi[:, :, c] = roi[:, :, c] * (1.0 - rotated_overlay[:, :, 3] / 255.0) + rotated_overlay[:, :, c] * (rotated_overlay[:, :, 3] / 255.0)
    face_image[y_offset:y_offset + rotated_overlay.shape[0], x_offset:x_offset + rotated_overlay.shape[1]] = roi

    return face_image

#ryan (lines 97-136)
def overlay_glasses(face_image, overlay, landmarks): # extra function so i can offset the glasses properly
    # landmarks of eyes
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]

    # center point calculation
    center_x = (left_eye[0][0] + right_eye[3][0]) // 2
    center_y = (left_eye[0][1] + right_eye[3][1]) // 2

    # Calculate the width of the overlayed image
    overlay_width = int(np.linalg.norm(right_eye[0] - left_eye[3]))

    # resizing
    overlay_resized = cv2.resize(overlay, (overlay_width*18, int(overlay_width / overlay.shape[1] * overlay.shape[0])*14))

    # position to offset the overlay image
    x_offset = center_x - (overlay_resized.shape[1] // 2) - 10
    y_offset = center_y - (overlay_resized.shape[0] // 2) + 17

    # to calculate the angle of rotation
    angle = np.arctan2(right_eye[3][1] - left_eye[0][1], right_eye[3][0] - left_eye[0][0]) * 180 / np.pi

    # to rotate the overlay
    rotation_matrix = cv2.getRotationMatrix2D((overlay_resized.shape[1] // 2, overlay_resized.shape[0] // 2), -angle, 1)
    rotated_overlay = cv2.warpAffine(overlay_resized, rotation_matrix, (overlay_resized.shape[1], overlay_resized.shape[0]))

    # calculate the ROI (region of interest) to put the overlay on
    roi = face_image[y_offset:y_offset + rotated_overlay.shape[0], x_offset:x_offset + rotated_overlay.shape[1]]

    # ensure that the shapes of the ROI and rotated overlay match
    if roi.shape[0] != rotated_overlay.shape[0] or roi.shape[1] != rotated_overlay.shape[1]:
        #print("Shapes of ROI and rotated sunglasses do not match.") #error got annoying so i removed it ez
        return face_image

    #combine the overlay image with the face image
    for c in range(0, 3):
        roi[:, :, c] = roi[:, :, c] * (1.0 - rotated_overlay[:, :, 3] / 255.0) + rotated_overlay[:, :, c] * (rotated_overlay[:, :, 3] / 255.0)
    face_image[y_offset:y_offset + rotated_overlay.shape[0], x_offset:x_offset + rotated_overlay.shape[1]] = roi

    return face_image


# Fancy UI time !!
#rian (lines 141-206)
print(" ____  _           _        ____              _   _")
print("|  _ \| |__   ___ | |_ ___ | __ )  ___   ___ | |_| |__")
print("| |_) | '_ \ / _ \| __/ _ \|  _ \ / _ \ / _ \| __| '_ \ ")
print("|  __/| | | | (_) | || (_) | |_) | (_) | (_) | |_| | | |")
print("|_|   |_| |_|\___/ \__\___/|____/ \___/ \___/ \__|_| |_|")
# For the people who are the reason why shampoo bottles have instructions
print("\nPro tip:\nPress ESC to exit the program\nPress SPACE to take a screenshot / photo!\nShow the peace sign for free balloons")
print(''.join("-" for i in range(100))) #lazy to manually type - out 100 times
# selection menu
print("Select the items you want (enter its corresponding number):")
print("[1] Round glasses")
print("[2] Rectangular glasses")
print("[3] Speech bubble ('SST Pride - Past, Present, Future!')")
print("[4] Speech bubble ('Honouring Our STEM Legacy!')")
print("[5] Speech bubble ('Yes - SST is THE school!!!')")
print("[6] Speech bubble ('Innovation and Progress since 2010!')")
print("[7] Speech bubble ('SST STAFF PRIDE!!!')")
print("[8] Speech bubble ('PROUD OF SST!!!')")
print("[9] Speech bubble ('The red, the blue, the grey and you! SST SST SST!!!')")
print("[10] Thought bubble ('Celebrating 15 Years of Excellence!')")
print("[11] SST Inc Hoodie")
print("[12] SST - Infineon partnership Hoodie")
print("[13] SST - SMU partnership Hoodie")
print("[14] PforSST Hoodie")
print("[15] Top Hat")
print("[16] Party Hat")
print("Enter 0 to confirm your choices")


# Get input choices
choices = []
selection = [
    "Round glasses",
    "Rectangular glasses",
    "Speech bubble ('SST Pride - Past, Present, Future!')",
    "Speech bubble ('Honouring Our STEM Legacy!')",
    "Speech bubble ('Yes - SST is THE school!!!')",
    "Speech bubble ('Innovation and Progress since 2010!')",
    "Speech bubble ('SST STAFF PRIDE!!!')",
    "Speech bubble ('PROUD OF SST!!!')",
    "Speech bubble ('The red, the blue, the grey and you! SST SST SST!!!')",
    "Thought bubble ('Celebrating 15 Years of Excellence!')",
    "SST Inc Hoodie",
    "SST - Infineon partnership Hoodie",
    "SST - SMU partnership Hoodie",
    "PforSST Hoodie",
    "Top Hat",
    "Party Hat",
]

# continuously prompt the user for input until a valid choice is made
while True:
    try: # exception if choice is not a number
        choice = int(input("> ")) #cursors are cool
        if 0 <= choice <=16: #exception if choice is out of range
            if choice != 0:
                choices.append(choice)
            else:
                break
        else:
            print("Please enter a value within range")
    except:
        print("Please try entering a valid whole number corresponding to your choice")

# Print selected choices (quality of life is great rn look at this)
print("You have selected:")
#ryan(lines 208-214)
if not choices:
    print("No props")
else:
    for i in range(len(choices)):
        print(selection[choices[i]-1])
#user instructions
print("Press spacebar to take a photo!\nPress ESC to exit the program.\n")

#rian (lines 217-277)
# Oh god this is gonna be so painful to comment ugh
#loop for continuous camera capture and image processing
glasses, hats, speechbubbles, hoodies = 0
while True:
        _, image = cap.read() #get image from camera
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # turn it into greyscale
        rects = detector(gray, 0) # detect face
        for (i, rect) in enumerate(rects): #finding landmarks
            # convert shape to numpy array
            shape = predictor(gray, rect)
            image = overlay_image(image, yesbubble, shape)
            shape = face_utils.shape_to_np(shape)
            # highly efficient algorithm to display only the things you selected
            # Assuming choices is a list of integers representing the selected options
            if choices.__contains__(1): # I
                if glasses == 0:
                    image = overlay_glasses(image, round_glasses, shape)
                    glasses = 1
                else:
                    while True:
                        print("You already have glasses equipped, are you sure you want to continue (y/n)?")
                        choice = input("> ").lower()
                        if choice in ["y", "n", "yes", "no"]:
                            if choice == "y" or choice == "yes":
                                image = overlay_glasses(image, round_glasses, shape)
                            else:
                                continue

            if choices.__contains__(2): # L
                if glasses == 0:
                    image = overlay_glasses(image, rectangular_glasses, shape)
                    glasses = 1
                else:
                    while True:
                        print("You already have glasses equipped. The glasses will overlap. Are you sure you want to continue (y/n)?")
                        choice = input("> ").lower()
                        if choice in ["y", "n", "yes", "no"]:
                            if choice == "y" or choice == "yes":
                                image = overlay_glasses(image, rectangular_glasses, shape)
                            else:
                                continue

            if choices.__contains__(3): # O
                if speechbubbles == 0:
                    image = overlay_image(image, pridebubble, shape)
                    speechbubbles = 1
                else:
                    while True:
                        print("You already have a speech bubble. The speech bubbles will overlap. Are you sure you want to continue (y/n)?")
                        choice = input("> ").lower()
                        if choice in ["y", "n", "yes", "no"]:
                            if choice == "y" or choice == "yes":
                                image = overlay_image(image, pridebubble, shape)
                            else:
                                continue

            if choices.__contains__(4): # V
                if speechbubbles == 0:
                    image = overlay_image(image, stembubble, shape)
                    speechbubbles = 1
                else:
                    while True:
                        print("You already have a speech bubble. The speech bubbles will overlap. Are you sure you want to continue (y/n)?")
                        choice = input("> ").lower()
                        if choice in ["y", "n", "yes", "no"]:
                            if choice == "y" or choice == "yes":
                                image = overlay_image(image, stembubble, shape)
                            else:
                                continue

            if choices.__contains__(5): # E
                if speechbubbles == 0:
                    image = overlay_image(image, yesbubble, shape)
                    speechbubbles = 1
                else:
                    while True:
                        print("You already have a speech bubble. The speech bubbles will overlap. Are you sure you want to continue (y/n)?")
                        choice = input("> ").lower()
                        if choice in ["y", "n", "yes", "no"]:
                            if choice == "y" or choice == "yes":
                                image = overlay_image(image, yesbubble, shape)
                            else:
                                continue

            if choices.__contains__(6):
                if speechbubbles == 0:
                    image = overlay_image(image, innovationbubble, shape)
                    speechbubbles = 1
                else:
                    while True:
                        print("You already have a speech bubble. The speech bubbles will overlap. Are you sure you want to continue (y/n)?")
                        choice = input("> ").lower()
                        if choice in ["y", "n", "yes", "no"]:
                            if choice == "y" or choice == "yes":
                                image = overlay_image(image, innovationbubble, shape)
                            else:
                                continue

            if choices.__contains__(7): # I
                if speechbubbles == 0:
                    image = overlay_image(image, staffpride, shape)
                    speechbubbles = 1
                else:
                    while True:
                        print("You already have a speech bubble. The speech bubbles will overlap. Are you sure you want to continue (y/n)?")
                        choice = input("> ").lower()
                        if choice in ["y", "n", "yes", "no"]:
                            if choice == "y" or choice == "yes":
                                image = overlay_image(image, staffpride, shape)
                            else:
                                continue

            if choices.__contains__(8): # F
                if speechbubbles == 0:
                    image = overlay_image(image, proudofsst, shape)
                    speechbubbles = 1
                else:
                    while True:
                        print("You already have a speech bubble. The speech bubbles will overlap. Are you sure you want to continue (y/n)?")
                        choice = input("> ").lower()
                        if choice in ["y", "n", "yes", "no"]:
                            if choice == "y" or choice == "yes":
                                image = overlay_image(image, proudofsst, shape)
                            else:
                                continue

            if choices.__contains__(9):
                if speechbubbles == 0:
                    image = overlay_image(image, redbluegrey, shape)
                    speechbubbles = 1
                else:
                    while True:
                        print("You already have a speech bubble. The speech bubbles will overlap. Are you sure you want to continue (y/n)?")
                        choice = input("> ").lower()
                        if choice in ["y", "n", "yes", "no"]:
                            if choice == "y" or choice == "yes":
                                image = overlay_image(image, redbluegrey, shape)
                            else:
                                continue

            if choices.__contains__(10): # E
                image = overlay_image(image, celebrating, shape) #thought bubble

            if choices.__contains__(11): # L
                if hoodies == 0:
                    image = overlay_image(image, sstinc, shape)
                    hoodies = 1
                else:
                    while True:
                        print("You already have a hoodie. Multiple hoodies will overlap. Are you sure you want to continue (y/n)?")
                        choice = input("> ").lower()
                        if choice in ["y", "n", "yes", "no"]:
                            if choice == "y" or choice == "yes":
                                image = overlay_image(image, sstinc, shape)
                            else:
                                continue

            if choices.__contains__(12): # S
                if hoodies == 0:
                    image = overlay_image(image, sst_infineon, shape)
                    hoodies = 1
                else:
                    while True:
                        print("You already have a hoodie. Multiple hoodies will overlap. Are you sure you want to continue (y/n)?")
                        choice = input("> ").lower()
                        if choice in ["y", "n", "yes", "no"]:
                            if choice == "y" or choice == "yes":
                                image = overlay_image(image, sst_infineon, shape)
                            else:
                                continue

            if choices.__contains__(13): # E
                if hoodies == 0:
                    image = overlay_image(image, sstsmu, shape)
                    hoodies = 1
                else:
                    while True:
                        print("You already have a hoodie. Multiple hoodies will overlap. Are you sure you want to continue (y/n)?")
                        choice = input("> ").lower()
                        if choice in ["y", "n", "yes", "no"]:
                            if choice == "y" or choice == "yes":
                                image = overlay_image(image, sstsmu, shape)
                            else:
                                continue

            if choices.__contains__(14):
                if hoodies == 0:
                    image = overlay_image(image, pforssst, shape)
                    hoodies = 1
                else:
                    while True:
                        print("You already have a hoodie. Multiple hoodies will overlap. Are you sure you want to continue (y/n)?")
                        choice = input("> ").lower()
                        if choice in ["y", "n", "yes", "no"]:
                            if choice == "y" or choice == "yes":
                                image = overlay_image(image, pforssst, shape)
                            else:
                                continue

            if choices.__contains__(15): # statements
                if hats == 0:
                    image = overlay_image(image, top_hat, shape)
                    hats = 1
                else:
                    while True:
                        print("You already have a hat. Multiple hats will overlap. Are you sure you want to continue (y/n)?")
                        choice = input("> ").lower()
                        if choice in ["y", "n", "yes", "no"]:
                            if choice == "y" or choice == "yes":
                                image = overlay_image(image, top_hat, shape)
                            else:
                                continue

            if choices.__contains__(16): # !!
                if hats == 0:
                    image = overlay_image(image, partyhat, shape)
                    hats = 1
                else:
                    while True:
                        print("You already have a hat. Multiple hats will overlap. Are you sure you want to continue (y/n)?")
                        choice = input("> ").lower()
                        if choice in ["y", "n", "yes", "no"]:
                            if choice == "y" or choice == "yes":
                                image = overlay_image(image, partyhat, shape)
                            else:
                                continue
                                
            ''' #ugly green circles for landmarks on face
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            '''

        # show output
        cv2.imshow("Output", image)

        k = cv2.waitKey(5) & 0xFF
        # if key press = ESC the program kills itself
        if k == 27:
            break
        # Taking photos!! I mean is a photo booth so what type of photobooth doesnt take photos like cmon
        # capture a photo, generate a timestamped filename, and save the image.
        if k == 32:
            current_time = datetime.datetime.now()
            # Fancy name formatting with DD/MM/YYYY_HH-MM-SS
#ryan(lines 279-287)
            timestamp = current_time.strftime("%d-%m-%Y_%H-%M-%S")
            savedFilename = "picture_" + timestamp + ".png"
            cv2.imwrite(savedFilename, image)
            print("Image saved!")
            print("Saved as: " + savedFilename)

#resource cleanup
cap.release()
cv2.destroyAllWindows()
