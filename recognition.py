from tkinter import *
import PIL.Image, PIL.ImageTk
from yolo import yolo
from model import predect
import cv2

time_start = 0
class App:
    def __init__(self, window, window_title, video_source):
        self.frames = []
        self.window = window
        self.window.title(window_title)
        self.vid = video_source
        self.canvas = Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
        self.delay = 1
        self.update()
        self.name=''

        self.window.mainloop()

    def update(self):
        exist, frame = self.vid.get_frame()
        if exist:  # if get the frame
            crop,box=yolo(frame)
            if box.__len__()!=0:
                (x,y,w,h)=box
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
                cv2.putText(frame, self.name, (x, y -10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
                self.frames.append(crop)
                if len(self.frames)==16:
                    self.name=predect(self.frames)
                    # self.name='001'
                    cv2.putText(frame, self.name, (x, y -10), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
                    self.frames.clear()

            self.frame = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.frame, anchor=NW)

        self.window.after(self.delay, self.update)



def main(video_source):
    App(Toplevel(), "GaitNet Recognition", video_source)