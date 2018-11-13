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

    absdiffFrame = cv2.absdiff(gray, backgroundGray)



    ret, th1 = cv2.threshold(absdiffFrame,150,255,cv2.THRESH_BINARY,cv2.THRESH_OTSU)

    edges = cv2.Canny(th1, 2, 2*2,3)

    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for i in contours:
         conArea = cv2.contourArea(i)
         if conArea > 2000 and conArea < 3500:
             diceBoundsRect = cv2.boundingRect(contours)
             cv2.rectangle(frame,diceBoundsRect.t1(),diceBoundsRect.br(),(0,0,255),2,8,0)


    cv2.imshow('video',frame)
    cv2.imshow('proccessing', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

