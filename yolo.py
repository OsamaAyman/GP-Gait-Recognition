import cv2
import numpy as np


# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
def yolo(img):
	classes = []
	with open("coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]
	layer_names = net.getLayerNames()
	output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))

	height, width, channels = img.shape
	# Detecting objects
	blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0, 0, 0), True, crop=False)
	# blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

	net.setInput(blob)
	outs = net.forward(output_layers)
	# Showing informations on the screen
	class_ids = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:

			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.1:
				# Object detected
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)

				# Rectangle coordinates
				x = int(center_x - w / 2)
				y = int(center_y - h / 2)

				boxes.append([x, y, w, h])
				confidences.append(float(confidence))
				class_ids.append(class_id)
	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
	a = 0
	b = 0
	if len(boxes)>0:
		for i in range(len(boxes)):
			if boxes[i][3] > a:
				a = boxes[i][3]
				b = i
		x = boxes[b][0]
		y = boxes[b][1]
		w = boxes[b][2]
		h = boxes[b][3]
		if x<0:x=0
		if y<0:y=0
		color = colors[b]
		img = img[y:y + h, x:x + w]
		return img,(x,y,w,h)
	else:
		return None,()

