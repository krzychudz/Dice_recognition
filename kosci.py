import cv2
import numpy


video = cv2.VideoCapture(0)


if video.isOpened() == False:
    print("Video Caputre problem")

ret, backgroundFrame = video.read()

cv2.imwrite('background.jpg', backgroundFrame)
backgroundGray = cv2.cvtColor(backgroundFrame, cv2.COLOR_BGR2GRAY)

while(True):
    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    absdiffFrame = cv2.absdiff(gray, backgroundGray)



    ret, th1 = cv2.threshold(absdiffFrame,150,255,cv2.THRESH_BINARY,cv2.THRESH_OTSU)

    edges = cv2.Canny(th1, 1, 3, 1)

    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in contours:
        conArea = cv2.contourArea(i)
        if conArea > 1500 and conArea < 15000:
            x,y,w,h = cv2.boundingRect(i)
            cv2.rectangle(frame,(x,y),(x+w, y+h),(0,0,255),2,8,0)
            kostka = frame[y:y+h, x:x+w]
            kostka = cv2.resize(kostka, (150,150))
            kostka = cv2.cvtColor(kostka, cv2.COLOR_BGR2GRAY)
            ret, kostka = cv2.threshold(kostka, 220, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
            h, w = kostka.shape[:2]
            mask = numpy.zeros((h + 2, w + 2), numpy.uint8)
            for j in [(0,0), (0,149), (149,0), (149,149)]:
                cv2.floodFill(kostka, mask, j,255)
            detector = cv2.SimpleBlobDetector(filterByInertia = True, minInertiaRatio = 0.5)
            #keypoints = detector.detect(kostka)
            #im_with_keypoints = cv2.drawKeypoints(kostka, keypoints, numpy.array([]), (0, 0, 255),
            #                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            #cv2.imshow("Keypoints", im_with_keypoints)
            cv2.imshow('dice', kostka)
    cv2.imshow('video',frame)
    cv2.imshow('proccessing', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

