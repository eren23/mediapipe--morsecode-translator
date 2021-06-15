import cv2
import mediapipe as mp
import time


def rescaleFrame( frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

cap = cv2.VideoCapture(0)

pTime=0

mp_draw = mp.solutions.drawing_utils
mp_facemesh = mp.solutions.face_mesh
facemesh = mp_facemesh.FaceMesh(max_num_faces=2)
draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=2)
    

ear1prev = []
ear2prev = []
wordArray = []
isLong = False
blinkedFor = 0
notBlinkedFor = 0
wasBlinked = False
letterArray = ""
letterIs =""


MorseCode = {
    "SL": "A",
    "LSSS":"B",
    "LSLS": "C",
    "LSS": "D",
    "S": "E",
    "SSLS": "F",
    "LLS": "G",
    "SSSS":"H",
    "SS": "I",
    "SLLL":"J",
    "LSL": "K",
    "SLSS": "L",
    "LL": "M",
    "LS": "N",
    "LLL": "O",
    "SLLS": "P",
    "LLSL": "Q",
    "SLS": "R",
    "SSS": "S",
    "L": "T",
    "SSL": "U",
    "SSSL": "V",
    "SLL": "W",
    "LSSL": "X",
    "LSLL": "Y",
    "LLSS": "Z" 
}


while True:
    success, img = cap.read()
    scaled = rescaleFrame(img,150)

    imgRGB = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
    results = facemesh.process(imgRGB)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    count = 0
    if results.multi_face_landmarks:
        for faceLm in results.multi_face_landmarks:
            mp_draw.draw_landmarks(scaled, faceLm, mp_facemesh.FACE_CONNECTIONS, draw_spec, draw_spec)

            ear1= abs((faceLm.landmark[160].x-faceLm.landmark[144].x)**2-(faceLm.landmark[160].y-faceLm.landmark[144].y)**2 )+abs((faceLm.landmark[158].x-faceLm.landmark[153].x)**2-(faceLm.landmark[158].y-faceLm.landmark[153].y)**2 )/abs(((faceLm.landmark[33].x-faceLm.landmark[133].x)**2)-((faceLm.landmark[33].y-faceLm.landmark[133].y)**2)) 
            ear2= abs((faceLm.landmark[385].x-faceLm.landmark[380].x)**2-(faceLm.landmark[385].y-faceLm.landmark[380].y)**2 )+abs((faceLm.landmark[387].x-faceLm.landmark[373].x)**2-(faceLm.landmark[387].y-faceLm.landmark[373].y)**2 )/abs(((faceLm.landmark[362].x-faceLm.landmark[263].x)**2)-((faceLm.landmark[362].y-faceLm.landmark[263].y)**2)) 

            if count>10:
                count=0
            else:
                count=count +1
          
            if len(ear1prev)>10:
                ear1prev[count] = ear1
                isLong = True
            else:
                ear1prev.append(ear1)
                

            if len(ear2prev)>10:
                ear2prev[count] = ear2
                isLong=True
            else:
                ear2prev.append(ear2)

            if isLong:
                if((ear1prev[abs(count-9)]*0.70>ear1) and (ear2prev[abs(count-9)]*0.70>ear2)):
                    print("blink")
                    wasBlinked = True
                    blinkedFor = blinkedFor+1
                else:
                    if(blinkedFor>(fps*.8)):
                        print("LONG blink")
                        letterArray = letterArray + "L"
                        blinkedFor =0
                        
                    elif(blinkedFor>int(fps/8)):
                        print("SHORT blink")
                        letterArray = letterArray + "S"
                        blinkedFor =0
                        
                    else:

                        notBlinkedFor = notBlinkedFor+1

                        print("no blink")
                        if(notBlinkedFor>fps*2):
                            print(letterArray)
                            if letterArray in MorseCode:
                                letterIs = MorseCode[letterArray]
                                wordArray.append(letterIs)
                                letterArray=""
                            letterArray=""
                            notBlinkedFor=0
                            

    finalW = ''.join(wordArray)
    if("ECE" in finalW):
        finalW = finalW + "<3"
    cv2.putText(scaled, str(int(fps)), (50,50),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),2)
    cv2.putText(scaled, str(finalW), (80,100),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
    cv2.imshow("Image", scaled)
    cv2.waitKey(1)