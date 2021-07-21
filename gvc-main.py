import cv2
import time
import math
import numpy as np
import handTrackingModule as htm
from imutil import FPS
from imutil import WebCamVideoStream

# pycaw - volume controller library
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def main():
    stream = WebCamVideoStream(src=0).start()  # Built-in primary webcam
    # stream = WebCamVideoStream(src='http://192.168.29.204:4747/mjpegfeed?640x480').start() # Droidcam
    fps = FPS().start() #starting fps
    det = htm.handDetector(detectCon=0.65)
    
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    vol = 0
    minVol, maxVol, _ = volume.GetVolumeRange()

    while True:
        img = stream.read()
        img2 = det.findHands(img, draw=False)
        lmk = det.findPosition(img2, draw=False)
        # Tips of index finger and thumb - id 8 and id 4 respectively (as per the landmark image)
        if len(lmk) != 0:
            _, x4, y4 = lmk[4]
            _, x8, y8 = lmk[8]

            cv2.circle(img, (x4, y4), 8, (255, 75, 0), cv2.FILLED)
            cv2.circle(img, (x8, y8), 8, (255, 75, 0), cv2.FILLED)
            cv2.line(img, (x4,y4), (x8,y8), (255, 75, 0), 2)

            # distance between the finger tips
            dist = math.hypot(x8-x4, y8-y4)
            
            # Hand movement range
                # min distance - 25 
                # max distance - 225
            # Volume range : -65 to 0
            vol = np.interp(dist, [25,200], [minVol, maxVol])
            volume.SetMasterVolumeLevel(vol, None)
        
        fps.update()
        fps.stop()
        # cv2.putText(img, str(int(fps.fps())), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 20, 240), 3)

        cv2.imshow("Camera", img)
        key = cv2.waitKey(1) & 0xFF ### get the last 8 bit of binary value
        # Check and change the key value below to change the exit key
        # Exit loop if q is pressed
        if key == ord('q'):
            break

    # Exit code !!!important
    cv2.destroyAllWindows()
    stream.stop()
    exit()

if __name__ == "__main__":
    main()


