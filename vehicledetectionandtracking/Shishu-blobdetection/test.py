import numpy as np
import time
import imutils
import cv2
import operator
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('dataset/video3.mp4')
# cap = cv2.VideoCapture('dataset/training.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
car_cascade = cv2.CascadeClassifier('cars.xml')
kernel = np.ones((3,3),np.uint8)
# erosion = cv2.erode(img,kernel,iterations = 1)


_,f = cap.read()
gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
frame = imutils.resize(gray, width=500)
weight = np.size(frame, 0)
# print(weight)
height = np.size(frame, 1)
accumulatematrix = np.zeros((weight, height, 256))
CAR_THRESHOLD=height/16
LOW_BOUND=np.uint16(height*0.3)
HIGH_BOUND=np.uint16(height*0.5)
rx=150
ry=20
CAR_TYPE=['car','trunk']
# background = np.zeros((weight-rx, height-2*ry))
background=np.zeros((weight,height))
diff=np.zeros((weight,height))
flag=0
threshold=10
haar_open=0
num_kp=0
num_haar=0

def Tracking_car():

    # Set up the detector with default parameters.
    detector = blob_initial()
    frame_count = 0
    while(1):
        ret, frame = cap.read()


        if frame_count%1==0:
            im = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            red=im[:,:,0]
            green=im[:,:,1]
            blue=im[:,:,2]
            # Detect blobs.
            if flag==0:
                fgmask_r = fgbg.apply(red)
                fgmask_g = fgbg.apply(green)
                fgmask_b = fgbg.apply(blue)
                ret, bred = cv2.threshold(fgmask_r, threshold, 255, cv2.THRESH_BINARY)
                ret, bgreen = cv2.threshold(fgmask_g, threshold, 255, cv2.THRESH_BINARY)
                ret, bblue = cv2.threshold(fgmask_b, threshold, 255, cv2.THRESH_BINARY)
                # if fgmask_b==fgmask_g:
                #     print('yes')
                # print(new)
                # sumrgb=np.add(fgmask_r,fgmask_g)
                # fgmask=np.add(sumrgb,fgmask_b)
                sumrgb=np.logical_or(bred,bgreen)
                fgmask=np.logical_or(sumrgb,bblue)
                fgmask=np.uint8(fgmask)*255
                blur = cv2.medianBlur(fgmask, 3)
                # fgmask = cv2.dilate(blur, None, iterations=3)
                fgmask = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
                # print('sum',np.sum(fgmask_r),np.sum(fgmask_g),np.sum(fgmask_b),np.sum(fgmask))
                shape = np.shape(fgmask)
                new = np.zeros(shape, np.uint8) + 255
                # blur = cv2.GaussianBlur(fgmask, (5, 5), 0)
                blur = np.subtract(new, blur)

            elif flag==1:
                if frame_count<101:
                    bg=bg_extractor(gray,frame_count,rx,ry,weight,height)
                diff=np.abs(np.subtract(gray,bg))

                ret, diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
                fgmask=cv2.convertScaleAbs(diff)
                bg_im=cv2.convertScaleAbs(bg)
                blur = cv2.medianBlur(fgmask, 5)


            keypoints = detector.detect(blur)
            num_kp = len(keypoints)
            carID=[]
            if num_kp is not 0:
                for idx in range(0,num_kp):
                    ptx=keypoints[idx].pt[0]
                    pty=keypoints[idx].pt[1]
                    rad=keypoints[idx].size/2
                    if pty>=LOW_BOUND and pty<=HIGH_BOUND:
                        cid=car_classification(blur,ptx,pty,rad)
                        carID.append(cid)
                        cv2.putText(im, '%s' % (CAR_TYPE[cid]),
                                    (np.uint16(ptx-rad), np.uint16(pty-rad)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 0), 1)
                        cv2.rectangle(im,(np.int16(ptx-rad),np.int16(pty-rad)),(np.int16(ptx+rad),np.int16(pty+rad)),(0,255,0),2)
                    else:
                        cv2.rectangle(im, (np.int16(ptx - rad), np.int16(pty - rad)),
                                      (np.int16(ptx + rad), np.int16(pty + rad)), (0, 0, 255), 1)
                    # print('blob:')
                    # print (ptx,pty)
                    # print (rad)
            cv2.line(im,(0,LOW_BOUND),(height,LOW_BOUND),(0,255,0),1)
            cv2.line(im,(0,HIGH_BOUND),(height,HIGH_BOUND),(0,255,0),1)

            # Draw detected blobs as red circles.
            # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
            # im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 255, 255),
            #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.putText(im, 'blob:%d' %(num_kp),
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 1)
            if haar_open == 1:
                cars = car_cascade.detectMultiScale(gray, 1.1, 1)
                num_haar = len(cars)
                for (x, y, w, h) in cars:
                    print('car_haarcascade:')
                    print(x, y)
                    print((w + h) / 2)
                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(im, 'haar:%d' % (num_haar),
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 1)
                # print(keypoints.pt[0])

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        try:

            # Show keypoints
            # plt.subplot(231), plt.imshow(im,'gray'), plt.title('Keypoints')
            # plt.subplot(232), plt.imshow(fgmask), plt.title('frame')
            # plt.subplot(233), plt.imshow(blur), plt.title('blur')
            # plt.subplot(234), plt.imshow(fgmask_r), plt.title('red')
            # plt.subplot(235), plt.imshow(fgmask_g), plt.title('green')
            # plt.subplot(236), plt.imshow(fgmask_b), plt.title('blue')
            # plt.show()
            rgb_frame=cv2.hconcat((fgmask_r,fgmask_g,fgmask_b))
            final_frame = cv2.hconcat((fgmask,blur))
            cv2.imshow('Keypoints', im)
            cv2.imshow('final',final_frame)
            cv2.imshow('rbg',rgb_frame)
            # cv2.imshow('frame', fgmask)
            # cv2.imshow('blur',blur)
            # cv2.imshow('red',fgmask_r)
            # cv2.imshow('green',fgmask_g)
            # cv2.imshow('blue',fgmask_b)
            # cv2.pause(100)
        except:
            print('error')

        # cv2.waitKey(50)
        frame_count+=1
    cap.release()
    cv2.destroyAllWindows()

def blob_initial():
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 255
    params.thresholdStep = 30

    # Change distance
    params.minDistBetweenBlobs = 3

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 300

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.3

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)
    return detector

def bg_extractor(frame,framecount,rx,ry,weight,height):
    for x in range(rx,weight):
        for y in range(ry,height-ry):
            z=frame[x,y]
            accumulatematrix[x,y,z]+=1
            if framecount%100==0:
                idx,value=max(enumerate(accumulatematrix[x,y,:]),key=operator.itemgetter(1))
                # background[x-rx,y-ry]=idx
                background[x,y]=idx

    # res = cv2.convertScaleAbs(background)
    return background


def car_classification(bw,cx,cy,radius):
    # x1=cx-radius
    # x2=cx+radius
    # y1=cy-radius
    # y2=cy+radius
    # pixels=sum(bw[x1:x2,y1:y2])
    if radius>=CAR_THRESHOLD:
        car_id=1
    else:
        car_id=0

    return car_id

if __name__ == '__main__':
    Tracking_car()