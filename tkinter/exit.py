#!/usr/bin/env python
# coding: utf-8

# In[1]:



import tkinter as tk
from tkinter import * 
from tkinter.ttk import *
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk
import cv2
import sys



full_screen=True

try:
    args1=sys.argv[1]
    if(args1=='d'):
        full_screen=False
        
except:
    full_screen=True


    
    
import numpy as np
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45
 
# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1
 
# Colors.
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)

classesFile = "ing.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

    
modelWeights = 'burger_feb17.onnx'
net = cv2.dnn.readNet(modelWeights)

def draw_label(im, label, x, y):
    """Draw text onto image at location."""
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle.
    cv2.rectangle(im, (x,y), (x + dim[0], y + dim[1] + baseline), (0,0,0), cv2.FILLED);
    # Display text inside the rectangle.
    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

def pre_process(input_image, net):
      # Create a 4D blob from a frame.
      blob = cv2.dnn.blobFromImage(input_image, 1/255,  (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)
 
      # Sets the input to the network.
      net.setInput(blob)
 
      # Run the forward pass to get output of the output layers.
      outputs = net.forward(net.getUnconnectedOutLayersNames())
      return outputs


def post_process(input_image, outputs):
      # Lists to hold respective values while unwrapping.
        
      class_ids = []
      confidences = []
      boxes = []
      # Rows.
      rows = outputs[0].shape[1]
      image_height, image_width = input_image.shape[:2]
      # Resizing factor.
      x_factor = image_width / INPUT_WIDTH
      y_factor =  image_height / INPUT_HEIGHT
      # Iterate through detections.
      for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]
            # Discard bad detections and continue.
            if confidence >= CONFIDENCE_THRESHOLD:
                  classes_scores = row[5:]
                  # Get the index of max class score.
                  class_id = np.argmax(classes_scores)
                  #  Continue if the class score is above threshold.
                  if (classes_scores[class_id] > SCORE_THRESHOLD):
                        confidences.append(confidence)
                        class_ids.append(class_id)
                        cx, cy, w, h = row[0], row[1], row[2], row[3]
                        left = int((cx - w/2) * x_factor)
                        top = int((cy - h/2) * y_factor)
                        width = int(w * x_factor)
                        height = int(h * y_factor)
                        box = np.array([left, top, width, height])
                        boxes.append(box)
                        
      indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
      for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]             
            # Draw bounding box.             
            cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
            # Class label.                      
            label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])             
            # Draw label.             
            draw_label(input_image, label, left, top)
            #print("Drawing label____________")
            ingredient_detected(label[0])
            #print(label)
      return input_image


 

root = tk.Tk()



win_width= int(root.winfo_screenwidth()*2/3)
win_height= int(root.winfo_screenheight()*2/3)
#setting tkinter window size

if(full_screen):
    win_width= int(root.winfo_screenwidth())
    win_height= int(root.winfo_screenheight())
    root.attributes('-fullscreen', True)
else:
    win_width= int(root.winfo_screenwidth()*2/3)
    win_height= int(root.winfo_screenheight()*2/3)
    
    
root.geometry("%dx%d" % (win_width, win_height))




tot_layers=5
c_width=525
c_height=500
w=400
pos_x=(c_width-w)/2
pos_y=c_height/8
border=10


vid_width=int(win_width - c_width - 4*border)
vid_height=int(win_height- 3*border)



root.bind("<Escape>", lambda event:root.destroy())



ingredients_collected=[]

keep_running=1


allowed_ingredients=['B', 'L', 'T', 'P', 'C', 'U']
style = 'BLPCB'
cols={
    'B':'peru',
    'L':'green',
    'T':'red',
    'P':'brown',
    'C':'yellow',
    'U':'sandybrown'
  }

#layer_names=[] #['L0', 'L1', 'L2', 'L3', 'L4']
#for i in range(tot_layers):
#    layer_names.append('L'+str(i))
#    print(i)

layer_names=[5,4,3,2,1,0] #['L0', 'L1', 'L2', 'L3', 'L4']

fill=0.9

ing_label=Label(root,text='Looking..')

def key_pressed(event):
    global ingredients_collected
    global ing_label
    ch=event.char
    if(ch in allowed_ingredients):
        if(ch not in ingredients_collected):
            ingredients_collected.append(ch)
    
    
    #ing_label=Label(root,)
    #canvas.itemconfig(ing_label, text=)
    ing_label.configure(text=','.join(ingredients_collected))
    change_all_colors()
    root.update()

    
    
def ingredient_detected(ch):
    global ingredients_collected
    global ing_label
    if(ch in allowed_ingredients):
        if(ch not in ingredients_collected):
            ingredients_collected.append(ch)
    
    
    #ing_label=Label(root,)
    #canvas.itemconfig(ing_label, text=)
    ing_label.configure(text=','.join(ingredients_collected))
    change_all_colors()
    root.update()

def shutdown():
    root.destroy()
    

new_burger_color='whitesmoke'

def change_color(it='r2', cx='gray'):
    canvas.itemconfig(it, fill=cx)

    
def change_all_colors():
    e=0
    for i in ingredients_collected:
            change_color(layer_names[e],cols[i] )
            e+=1

def new_burger(l):
    global ingredients_collected
    ingredients_collected=[]
    for i in range(l):
        change_color(layer_names[i],new_burger_color )
    

def complete_burger(ca=None, cb=None):
    global keep_running
    keep_running=0
    print ("STOPPING")

    
b_LPC = PhotoImage(file = "button_images/b-LPC.png")
b_LPCT = PhotoImage(file = "button_images/b-LPCT.png")
b_PCTL = PhotoImage(file = "button_images/b-PCTL.png")
b_PL = PhotoImage(file = "button_images/b-PL.png")
b_PT = PhotoImage(file = "button_images/b-PT.png")

complete = PhotoImage(file = "button_images/complete.png")
#b_LPC = b_LPC.subsample(1, 1)

shut = PhotoImage(file = "button_images/shutdown.png")

canvas = tk.Canvas(root, width=c_width, height=c_height, bg='white')

cw=c_height/tot_layers


#LAYER-1
layer=1
col='white'
outline_color='white'
L4 = canvas.create_arc(pos_x, cw*(layer-1)+2*cw*(1-fill)/2, pos_x+w, cw*(layer-1)+2*cw*(1-fill)/2+2*cw*fill-cw*(1-fill), start=0, extent=180, fill=col, style='chord', outline=outline_color)

layer=2
#col='gray'
L3 = canvas.create_rectangle(pos_x, cw*(layer-1)+cw*(1-fill)/2, pos_x+w, cw*(layer-1)+cw*(1-fill)/2+cw*fill, fill=col, outline=outline_color)

#LAYER-3
layer=3
#col='green'
L2 = canvas.create_rectangle(pos_x, cw*(layer-1)+cw*(1-fill)/2, pos_x+w, cw*(layer-1)+cw*(1-fill)/2+cw*fill, fill=col, outline=outline_color)

#LAYER-4
layer=4
#col='pink'
#r2 = canvas.create_rectangle(pos_x, f*165, pos_x+w, f*175, fill="green")
L1 = canvas.create_rectangle(pos_x, cw*(layer-1)+cw*(1-fill)/2, pos_x+w, cw*(layer-1)+cw*(1-fill)/2+cw*fill, fill=col, outline=outline_color)

#LAYER-5
layer=5
#col='yellow'
L0 = canvas.create_arc(pos_x, cw*(layer-1)+2*cw*(1-fill)/2-cw, pos_x+w, cw*(layer-1)+cw*(1-fill)/2+cw*fill, start=180, extent=180, fill=col, outline=outline_color)

#canvas.create_arc(pos_x, 125, 180, 165, start=0, extent=180, fill="brown", style='chord')

b1=tk.Button(root, image=b_LPC, compound=CENTER, command=lambda: new_burger(5), width=100, height=100)
b2=tk.Button(root, image=b_LPCT, compound=CENTER, command=lambda: new_burger(4), width=100, height=100)
b3=tk.Button(root, image=b_PCTL, compound=CENTER, command=lambda: change_color(r2,'black'), width=100, height=100)
b4=tk.Button(root, image=b_PL, compound=CENTER, command=lambda: change_color(r3,'black'), width=100, height=100)
b5=tk.Button(root, image=b_PT, compound=CENTER, command=lambda: change_color(bot_bun,'black'), width=100, height=100)


b6=tk.Button(root, image=complete, compound=CENTER, command=lambda: complete_burger('xxxx','xxxxxx'))

b7=tk.Button(root, image=shut, compound=CENTER, command=lambda: shutdown(), width=100, height=100)

video = Label(root)


current_value=45
sca = Scale(root, from_=0, to=100, orient='vertical', variable=current_value)

slider_label = ttk.Label(
    root,
    text='Slider:'
)

b1.grid(row=1,column=0)
b2.grid(row=1,column=1)
b3.grid(row=1,column=2)
b4.grid(row=1,column=3)
b5.grid(row=1,column=4)
b6.grid(row=3,column=0, columnspan=3, sticky=W)
b7.grid(row=3,column=4, columnspan=1)
canvas.grid(row=2, column=0, columnspan=5,pady=2,padx=2)
ing_label.grid(row=3, column=3,columnspan=1, sticky=W )
sca.grid(row=3,column=2,sticky=E )
slider_label.grid(row=4,column=2,sticky='' )
    
video.grid(row=1,column=5,rowspan=5,pady=border,padx=border, sticky='')

root.bind('<Key>',key_pressed)


#frame=np.random.randint(0,255,[100,100,3],dtype='uint8')
#img = ImageTk.PhotoImage(Image.fromarray(frame))

#img=img.resize(iw, ih)






vid = cv2.VideoCapture(0)
while(True):
    if(keep_running==0):
        break

    ret, frame = vid.read() #Reads the video
    
    
    detections = pre_process(frame, net)
    detected_frame = post_process(frame.copy(), detections)
    
    
    #Converting the video for Tkinter
    detected_frame = cv2.resize(detected_frame, (vid_width,vid_height), interpolation = cv2.INTER_AREA)
    
    cv2image = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGBA)
    #cv2image = detected_frame#cv2.cvtColor(, cv2.COLOR_BGR2RGBA)
    
    img = Image.fromarray(cv2image)

    imgtk = ImageTk.PhotoImage(image=img)
    
    #Setting the image on the label
    video.config(image=imgtk)
    root.update() #Updates the Tkinter window

    
    
vid.release()
root.mainloop()


# In[ ]:




