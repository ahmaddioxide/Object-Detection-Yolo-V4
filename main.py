import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)



# OpenCV DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/200)

#Loadinig class list
classes=[]
with open("dnn_model/classes.txt","r") as file_object:

    for class_name in file_object.readlines():
        # print(class_name)
        class_name=class_name.strip()
        classes.append(class_name)

print("Classe we have in Model")
print(classes)

while True:
    # Getting frames
    ret, frame = cap.read()

    # Object detection
    (class_ids, scores, bboxes) = model.detect(frame)

    for class_id, score, bbox in zip(class_ids,scores,bboxes):
        (x,y,w,h)=bbox

        cv2.putText(frame,str(classes[class_id]),(x,y-10),cv2.FONT_HERSHEY_PLAIN,1,(200,0,60),2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(200,0,50),3)

    print("class ids", class_ids)
    print("score", scores)
    print('Boundary Boxes', bboxes)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
