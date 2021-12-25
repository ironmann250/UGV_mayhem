from djitellopy import tello
import numpy as np
import cv2,sys
import time
sys.path.insert(0, './../')


def thresRed(img):
    """for thresholding the red color from the color image"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    mask = mask1 | mask2
    return mask

def get_mask(img,imgHsv,lower,upper): #returns mask for given values
    mask = cv2.inRange(imgHsv,lower,upper)
    result = cv2.bitwise_and(img,img, mask = mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 
    return mask  

def getContours(img,imgContour):
    '''gets contours and returns biggest area's center and area and adds texts to the output image'''
    myObjectListData = []
    myObjectListC = []
    myObjectListArea = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin =30# cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            #print(len(approx))
            x , y , w, h = cv2.boundingRect(approx)
            cx = x + w // 2
            cy = y + h // 2
            area = w * h
            
            myObjectListC.append((cx, cy))
            myObjectListArea.append(area)
            myObjectListData.append([x,y,w,h,approx])
    if len(myObjectListArea) > 0:
        i = myObjectListArea.index(max(myObjectListArea))
        cv2.circle(imgContour, myObjectListC[i], 5, (0, 255, 0), cv2.FILLED)
        x,y,w,h,approx=myObjectListData[i]
        #print(myObjectListArea[i])
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
        return imgContour,[[0,0],0]


def trackObj(me, info, w,h, pidSpeed, pErrorSpeed,pidUd, pErrorUd,imgContour):
    '''tracking objects using errors in object centers and area also adjusting the height'''
    global change,timeWaited,waitTime
    area = info[1]
    x, y = info[0]
    fb = 0
    curr_speed=front_speed
    #calculate centering errors
    errorSpeed = x - w // 2
    errorUd = y - h // 2
    #calculate pid values for yaw
    speed = pidSpeed[0] * errorSpeed + pidSpeed[1] * (errorSpeed - pErrorSpeed)
    speed = int(np.clip(speed, -100, 100))
    #calculate pid values for up/down (not used anymore)
    ud = pidUd[0] * errorUd + pidUd[1] * (errorUd - pErrorUd)
    ud = int(np.clip(ud, -20, 20))

    #if in speeduprange speedup forward speed
    if area <= speedupRange[1]:
        if area >= speedupRange[0]:
            curr_speed=speedup_vel

    #calc front speed
    if area > fbRange[0] and area < fbRange[1]: 
        fb = 0
    if area > fbRange[1]:
        fb = -curr_speed
    elif area < fbRange[0] and area > 0:
        fb = curr_speed 
    #maintain height
    if abs(tello.get_height()-startHeight)>Herror:
        if tello.get_height() > startHeight:
            ud=-20
        else:
            ud=20
    else:
        ud=0
    #if no object stop drone
    if x == 0:
        timeWaited=time.time()-change
        if True:#timeWaited>waitTime:
            timeWaited=0
            speed = 0
            fb=0
            ud=0
            errorUd = 0
            errorSpeed = 0
    else:
        #removed timing logic when object suddenly disappear
        change=time.time()
        timeWaited=0

    #print(speed, fb)
    cv2.putText(imgContour, "LR: "+str(speed)+" FB: "+str(fb)+" UD: "+str(-ud),( 5, 200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2,(0, 0, 255), 2)
    cv2.putText(imgContour, "err LR: "+str(errorSpeed)+" err UD: "+str(x)+" tm wtd: "+str(timeWaited),( 5, 220), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2,(0, 0, 255), 2)
    #time.sleep(0.5)
    if not debug:
        me.send_rc_control(0, fb, ud, speed)
    return [errorSpeed,errorUd]
  
def is_target_detected(tello,samplings=240, ratio=0.65):
    ccnt=0
    for i in range(samplings):
        myFrame = tello.get_frame_read().frame
        img = cv2.resize(myFrame, (w, h))
        imgContour = img.copy()
        imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
        mask = cv2.inRange(imgHsv,lower,upper)
        result = cv2.bitwise_and(img,img, mask = mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        imgBlur = cv2.GaussianBlur(result, (7, 7), 1)
        imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
        threshold1 = 101
        threshold2 = 0
        imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
        kernel = np.ones((5, 5))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
        img, info = getContours(imgDil , imgContour) 
        if info[1]>0:
            cx,cy=info[0]
            if cx<(2*w/3) and cx > (w/3):
                ccnt+=1
        cv2.imshow('frame',img)
        cv2.waitKey(10)
    print(ccnt/samplings)

    return (ccnt/samplings>=ratio)



startHeight,Herror=[80,2]
debug=False #debug mode, uses computer camera not the drone
testTime=60 #auto shutdown drone after these seconds
waittime=0.5
change=0
timeWaited=0
multiplier=2 #increase the size of the output video
w, h = [360*multiplier, 240*multiplier]
frameWidth,frameHeight,deadZone=w,h,50

fbRange = [15000*(multiplier*multiplier),30000*(multiplier*multiplier)]#the range of areas at which to the drone stops
speedupRange = [2000*(multiplier*multiplier),10000*(multiplier*multiplier)]#the range of areas at which the drone speed's up
front_speed=30 #fnormal front speed
speedup_vel=50 #speed when in speeduprange
pidSpeed = [0.4, 0.4, 0]#yaw pid values
pErrorSpeed = 0
pidUd = [0.4, 0.4, 0]#up/down pid values (not used anymore )
pErrorUd = 0
#used in end of mission analysis (not used anymore)
endTargetCount=0
endTargetLimits=10
#[0,82,37,2,255,255]
#[0,255,0,0,255,176]
#[81,196,115,179,255,255]
upper = np.array([0,255,255])#red
lower = np.array([0,255,13])#blue outdoors h_min,s_min,v_min

tello = tello.Tello() # initialize tello 
tello.connect()
tello.streamon() 



while(1):
    #myFrame = tello.get_frame_read().frame
    #print("battery level is {}".format(tello.get_battery()))
    #print("Reached tracking height of: {} cm".format(tello.get_height()))
    print (is_target_detected(tello))
    #cv2.imshow('dbdfb',myFrame)
    #cv2.waitKey(10)

print("battery level is {}".format(tello.get_battery()))
myFrame = tello.get_frame_read().frame
cv2.resize(myFrame, (w, h))

tello.takeoff()
# diff=startHeight-tello.get_height()
# if (diff) > 0:
#     go_to_height_v = 20
#     tello.move_up(diff)
# else:
#     go_to_height_v = -20
#     tello.move_down(-diff)
print("Reached tracking height of: {} cm".format(tello.get_height()))


now=time.time()
if debug:
    cap = cv2.VideoCapture(0)



while(0):
    #myFrame = tello.get_frame_read().frame
    #print("battery level is {}".format(tello.get_battery()))
    #print("Reached tracking height of: {} cm".format(tello.get_height()))
    is_target_detected(tello)
    #cv2.imshow('dbdfb',myFrame)
    #cv2.waitKey(10)

for i in range(3):
    tello.move_right(70)
    if is_target_detected(tello):
        break

locfile=open('location.txt','w')
locfile.write(str(i+1))
locfile.close()
tello.move_back(0)
tello.land()



# while True:
#     if testTime !=0 and (time.time()-now >=testTime):
#             tello.land()
#             break
#     if debug:
#         _, img = cap.read()
#         print(len(img))
#         img = cv2.resize(img, (w, h))
#     else:
#         myFrame = tello.get_frame_read().frame
#         img = cv2.resize(myFrame, (w, h))
#     imgContour = img.copy()
#     imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    
#     mask = cv2.inRange(imgHsv,lower,upper)

#     result = cv2.bitwise_and(img,img, mask = mask)
#     mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

#     imgBlur = cv2.GaussianBlur(result, (7, 7), 1)
#     imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
#     threshold1 = 101
#     threshold2 = 0
#     imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
#     kernel = np.ones((5, 5))
#     imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
#     img, info = getContours(imgDil , imgContour)
#     pErrorSpeed,pErrorUd = trackObj(tello, info, w,h, pidSpeed, pErrorSpeed,pidUd,pErrorUd,imgContour)
#     #nonPIDtracking(tello, info, w,h,imgContour)
#     #if isEndMission(img,now):
#     #    tello.land()
#     #    break

#     #print("Area", info[1], "Center", info[1])
#     cv2.imshow("output", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         tello.land()
#         break
