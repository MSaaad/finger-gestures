import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise

bg = None

def run_avg(image, accumWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, accumWeight)

def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        cv2.putText(clone, str('Put fingers in the box'), (70, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)
def count(thresholded, segmented):
    chull = cv2.convexHull(segmented)
    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    radius = int(0.8 * maximum_distance)
    circumference = (2 * np.pi * radius)
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
    (_, cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = 0
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1
        if count == 2:
            video1 = cv2.VideoCapture('1.mp4')
            if (video1.isOpened()== False): 
                print("Error opening video1 stream or file")
            while(video1.isOpened()):
                retur, fm = video1.read()
                if retur == True:
                    cv2.imshow('Frame',fm)
                    if cv2.waitKey(30) & 0xFF == ord('k'):
                      break
                else:
                    break
        if count == 3:
            video2 = cv2.VideoCapture('2.mp4')
            if (video2.isOpened()== False): 
                print("Error opening video2 stream or file")
            while(video2.isOpened()):
                retur, fm = video2.read()
                if retur == True:
                    cv2.imshow('Frame',fm)
                    if cv2.waitKey(30) & 0xFF == ord('k'):
                        break
                else:
                    break
        if count == 4:
            video3 = cv2.VideoCapture('3.mp4')
            if (video3.isOpened()== False): 
                print("Error opening video3 stream or file")
            while(video3.isOpened()):
                retur, fm = video3.read()
                if retur == True:
                    cv2.imshow('Frame',fm)
                    if cv2.waitKey(30) & 0xFF == ord('k'):
                        break
                else:
                    break
        if count == 5:
            video4 = cv2.VideoCapture('4.mp4')
            if (video4.isOpened()== False): 
                print("Error opening video4 stream or file")
            while(video4.isOpened()):
                retur, fm = video4.read()
                if retur == True:
                    cv2.imshow('Frame',fm)
                    if cv2.waitKey(30) & 0xFF == ord('k'):
                        break
                else:
                    break
    return count

if __name__ == "__main__":
    accumWeight = 0.5
    url = ('https://192.168.1.103:8080/video')
    camera = cv2.VideoCapture(url)
    top, right, bottom, left = 10, 350, 225, 590
    num_frames = 0
    calibrated = False
    while(True):
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print("[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] calibration successfull...")
        else:
            hand = segment(gray)
            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                fingers = count(thresholded, segmented)
                cv2.putText(clone, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow("Thesholded", thresholded)
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        num_frames += 1
        cv2.imshow("Video Feed", clone)
        q = cv2.waitKey(20)
        if q == ord("q"):
            break
camera.release()
cv2.destroyAllWindows()
