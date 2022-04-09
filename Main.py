from tkinter import *
import tkinter.font as tkFont
from tkinter.filedialog import askopenfile
import cv2
from helper import center
import numpy as np
from PIL import Image
import recognition

input_width = 540
input_height = 960


class VideoCapture:
    def __init__(self, video_source):
        self.resolution = (input_width,input_height)

        self.vid = cv2.VideoCapture(video_source)
        self.vid.set(3, self.resolution[1])
        self.vid.set(4, self.resolution[0])

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source, please check your camera settings", video_source)


        self.width = self.resolution[1]
        self.height = self.resolution[0]
        self.cnt = 0

    def get_frame(self): # simulation from video files
        if self.vid.isOpened():
            self.cnt+=1
            ret, frame = self.vid.read()
            if self.cnt<=1:
                return (None, None)
            else:
                self.cnt=0
            if ret:
                frame = np.array(Image.fromarray(frame).resize((self.resolution[1], self.resolution[0]), Image.ANTIALIAS))
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                self.vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return (None, None)





def enter():
    if chooseSource.get() == 1:
        video = VideoCapture(0)
    else:
        file = askopenfile(mode='r', filetypes=[('videos', '*.mp4'), ('videos', '*.avi')])
        video = VideoCapture(file.name)

    recognition.main(video)

if __name__ == '__main__':
    root =Tk()
    root.title("Gait Recognition")
    root.geometry("300x200")
    #fontStyle =tkFont.Font(family="Lucida Grande", size=20)
    radioGroup = LabelFrame(root, text='select source')
    radioGroup.pack(side=TOP, expand=YES)
    chooseSource = IntVar()
    Radiobutton(radioGroup, text="Upload video", variable=chooseSource, value=0).pack(anchor=W)
    Radiobutton(radioGroup, text="open Live camera", variable=chooseSource, value=1).pack(anchor=W)

    Button(root, text="....Enter...", command=enter).pack(side=BOTTOM)

    center(root)
    root.mainloop()