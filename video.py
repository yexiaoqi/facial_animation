import cv2
import dlib
import numpy as np
from skimage.transform import rescale

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

while (1):
    # get a frame
    ret, img = cap.read()

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    image = rescale(img, 0.5)
    image = (image * 255).astype(np.uint8)
    dets = detector(image, 1)
    #    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        #        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #           k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        temp_d = dlib.rectangle(d.left() * 2, d.top() * 2, d.right() * 2, d.bottom() * 2)
        shape = predictor(img, temp_d)
    #        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
    #                                                  shape.part(1)))
    # Draw the face landmarks on the screen.
    #    d=dets[0]
    #    shape = predictor(img, d)
    for i in range(0, 68):
        cv2.circle(img, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), -1)

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
