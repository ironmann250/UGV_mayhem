import cv2
import numpy as np
from robomaster import robot
from robomaster import camera
import time

frameWidth = 640
w=frameWidth
frameHeight = 480
h=frameHeight


deadZone=100
global imgContour

debug=False
testTime=0
waitTime=0.5
change=0
timeWaited=0
if not debug:
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera = ep_robot.camera
    ep_camera.start_video_stream(display=False, resolution='480p')
    ep_chassis = ep_robot.chassis
else:
    me=""
multiplier=1
w, h = frameWidth*multiplier, frameHeight*multiplier
frameWidth,frameHeight,deadZone=w,h,50
fbRange = [4900*(multiplier*multiplier),6000*(multiplier*multiplier)]#[6200, 6800]
pidSpeed = [0.4, 0.4, 0]
pErrorSpeed = 0
pidUd = [0.4, 0.4, 0]
pErrorUd = 0

def calculateHeightAndBeta(radius, center, inches):
    KNOWN_WIDTH = 6.5
    KNOWN_DIS = 32
    KNOWN_HEIGHT = 21.5
    KNOWN_RADIUS = 35
    KNOWN_INCHES = 40.83

    Z_1_BA = KNOWN_WIDTH / (KNOWN_RADIUS * 2) * 44
    BETA = np.arcsin(Z_1_BA / KNOWN_INCHES)
    ALPHA = np.arctan(KNOWN_HEIGHT / KNOWN_DIS) - BETA

    beta = np.arcsin((KNOWN_WIDTH * (center[1] - 240)) / (2 * radius * inches))
    height = inches * np.sin(ALPHA + beta)
    print("height:", height)

    height_real = (65 - 0) / (20.2258 - 27.9855) * (height - 27.9855)

    return height_real

def grab(ep_robot, radius, center, inches):
    ep_chassis = ep_robot.chassis
    ep_arm = ep_robot.robotic_arm
    ep_gripper = ep_robot.gripper

    real_height = calculateHeightAndBeta(radius, center, inches)-3
    print("real_height:", real_height)

    ep_gripper.open()
    time.sleep(3)
    ep_arm.moveto(x=150, y=real_height).wait_for_completed()
    ep_chassis.drive_speed(x=0.1, y=0, z=0, timeout=1)
    time.sleep(0.2)
    ep_gripper.close()
    time.sleep(3)
    


def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth

def getContours(img,imgContour):
    myObjectListData = []
    myObjectListC = []
    myObjectListArea = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin =200# cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            #print(len(approx))
            x , y , w, h = cv2.boundingRect(approx)
            cx = x + w // 2
            cy = y + h // 2
            area = w * h
            ((x_, y_), radius) = cv2.minEnclosingCircle(cnt)
            myObjectListC.append((cx,cy,radius))
            myObjectListArea.append((area))
            myObjectListData.append([x,y,w,h,approx])
    if len(myObjectListArea) > 0:
        i = myObjectListArea.index(max(myObjectListArea))

        cv2.circle(imgContour, (myObjectListC[i][0],myObjectListC[i][1]), 5, (0, 255, 0), cv2.FILLED)
        x,y,w,h,approx=myObjectListData[i]
        cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 5)

        cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                    (0, 255, 0), 2)
        cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.putText(imgContour, " " + str(int(x))+ " "+str(int(y)), (x - 20, y- 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                    (0, 255, 0), 2)

        cx = int(x + (w / 2))
        cy = int(y + (h / 2))

        if (cx <int(frameWidth/2)-deadZone):
            cv2.putText(imgContour, " GO LEFT " , (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 1)
           
        elif (cx > int(frameWidth / 2) + deadZone):
            cv2.putText(imgContour, " GO RIGHT ", (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 1)
         
        elif (cy < int(frameHeight / 2) - deadZone):
            cv2.putText(imgContour, " GO UP ", (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 1)
           
        elif (cy > int(frameHeight / 2) + deadZone):
            cv2.putText(imgContour, " GO DOWN ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 255), 1)
        
        cv2.line(imgContour, (int(frameWidth/2),int(frameHeight/2)), (cx,cy),
                (0, 0, 255), 3)

        return imgContour, [myObjectListC[i], myObjectListArea[i]]
    else:
        return imgContour,[[0,0,0],0]
        
    

def speedPID(sp, pv, pError):
    """
    determine speed as target is approached
    :param sp:  set point
    :param pv: processed value
    :return: speed
    """
    error = pv - sp
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))
    pError = error

    return speed


def findFace(img):
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray)

    #print(faces)

    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)
    if len(myFaceListArea) > 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]


def trackObj(me, info, w,h, pidSpeed, pErrorSpeed,pidUd, pErrorUd):
    global change,timeWaited,waitTime
    area = info[1]
    x, y,radius = info[0]
    fb = 0

    errorSpeed = x - w // 2
    errorUd = y - h // 2
    speed = pidSpeed[0] * errorSpeed + pidSpeed[1] * (errorSpeed - pErrorSpeed)
    speed = int(np.clip(speed, -100, 100))
    ud = pidUd[0] * errorUd + pidUd[1] * (errorUd - pErrorUd)
    ud = int(np.clip(ud, -20, 20))
    
    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    if area > fbRange[1]:
        fb = -0.1
    elif area < fbRange[0] and area > 0:
        fb = 0.1
    
    if x == 0:
        speed = 0
        fb=0
        ud=0
        errorUd = 0
        errorSpeed = 0

    #print(speed, fb)
    cv2.putText(imgContour, "LR: "+str(speed)+" FB: "+str(fb)+" UD: "+str(-ud),( 5, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2,(0, 0, 255), 2)
    cv2.putText(imgContour, "err LR: "+str(errorSpeed)+" err UD: "+str(x)+" tm wtd: "+str(timeWaited),( 5, 220), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2,(0, 0, 255), 2)
    #time.sleep(0.5)
    if not debug:
        ep_chassis.drive_speed(x=fb, y=0, z=speed, timeout=1)
        pass
    return errorSpeed,errorUd,fb

if debug:
    cap = cv2.VideoCapture(0)

now=time.time()
fbcount=0
fbsum=0
ffb=1
while True:
    if testTime !=0 and (time.time()-now >=testTime):
        if not debug:
            pass 
        break
    if debug:
        _, img = cap.read()
        img = cv2.resize(img, (w, h))
    else:
        img = ep_camera.read_cv2_image(timeout=3, strategy='newest')
        img = cv2.resize(img, (w, h))
    imgContour = img.copy()
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    #lower = np.array([137,80,180])#h_min,s_min,v_min
    #upper = np.array([179,255,255])#h_max,s_max,v_max
    #lower = np.array([89,154,83])#blue on drone h_min,s_min,v_min
    #upper = np.array([179,255,255])#blue on drone h_max,s_max,v_max
    #lower = np.array([45, 95, 40])#green ball h_min,s_min,v_min
    #upper = np.array([82, 240, 255])#green ball h_max,s_max,v_max
    lower = np.array([0,100,100])#red ball
    upper = np.array([10,255,255])#red ball
    #lower = np.array([36,156,48])#blue outdoors h_min,s_min,v_min
    #upper = np.array([133,255,255])#blue outdoors h_max,s_max,v_max
    
    mask = cv2.inRange(imgHsv,lower,upper)
    result = cv2.bitwise_and(img,img, mask = mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
 
    imgBlur = cv2.GaussianBlur(result, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    threshold1 = 0
    threshold2 = 27
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=2)
    #imgrod=cv2.erode(imgCanny,kernel,iterations=1)
    
    img, info = getContours(imgDil , imgContour)
    #fbcount=80
    if ffb==0:
        fbcount+=1
    if fbcount<100:
        pErrorSpeed,pErrorUd,ffb = trackObj(ep_chassis, info, w,h, pidSpeed, pErrorSpeed,pidUd,pErrorUd)
    else:
        radius=info[1]
        centerx,centery, radius =info[0]
        center=(centerx, centery)
        KNOWN_DISTANCE = 100
        KNOWN_WIDTH = 6.5
        focalLength = (2 * 14.3 * KNOWN_DISTANCE) / KNOWN_WIDTH
        dist = distance_to_camera(KNOWN_WIDTH, focalLength, 2 * radius)
        grab(ep_robot, radius, center, dist)
        ep_chassis.drive_speed(x=-0.5, y=0, z=0, timeout=1)
        time.sleep(3)
        ep_robot.robotic_arm.moveto(x=70, y=40).wait_for_completed()
        ep_robot.gripper.open()
        time.sleep(2)
        ep_chassis.drive_speed(x=-0.1, y=0, z=0, timeout=1)
        time.sleep(3)
        ep_robot.gripper.close()
        time.sleep(1)
        break
    #print("Area", info[1], "Center", info[1])
    cv2.imshow("output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        if not debug:
            pass
           # me.land()
        break

