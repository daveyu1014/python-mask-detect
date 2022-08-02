import cv2
import numpy as np
from matplotlib import pyplot as plt

net = cv2.dnn.readNetFromDarknet("cfg/yolov3-tiny-obj.cfg","yolov3-tiny-obj_30000.weights")  #讀模型
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = [line.strip() for line in open("obj.names")]
colors = [(0,0,255), (255,0,0), (0,255,0)]

img = cv2.imread("test01.jpeg")  #讀圖片
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape 
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)
class_ids = []
confidences = []
boxes = []
for out in outs:
    print(len(outs))
    for detection in out:
        tx, ty, tw, th, confidence = detection[0:5]
        scores = detection[5:]
        class_id = np.argmax(scores)  
        if confidence > 0.3:   
            center_x = int(tx * width)
            center_y = int(ty * height)
            w = int(tw * width)
            h = int(th * height)
            # 取得箱子方框座標
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
plt.rcParams['figure.figsize'] = [15, 10]
#img_rgb = cv2.cvtColor(img,)
cv2.imshow("test", img)
cv2.waitKey()
cv2.imwrite("cv2.jpg", img)
#plt.imshow(img_rgb)


