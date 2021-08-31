# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:28:11 2021

@author: sk
"""
#importing the required libraries
import cv2
import face_recognition
import numpy as np

#capture the video from default camera 
webcam_video_stream = cv2.VideoCapture(0)

#load the sample images and get the 128 face embeddings from them
modi_image = face_recognition.load_image_file('images/samples/modi.jpg')
modi_face_encodings = face_recognition.face_encodings(modi_image)[0]

trump_image = face_recognition.load_image_file('images/samples/trump.jpg')
trump_face_encodings = face_recognition.face_encodings(trump_image)[0]

abhi_image = face_recognition.load_image_file('images/samples/abhi.jpg')
abhi_face_encodings = face_recognition.face_encodings(abhi_image)[0]

zeynep_image = face_recognition.load_image_file('images/samples/zeynep.jpg')
zeynep_face_encodings = face_recognition.face_encodings(zeynep_image)[0]

serdar_image = face_recognition.load_image_file('images/samples/serdar.jpg')
serdar_face_encodings = face_recognition.face_encodings(serdar_image)[0]

#save the encodings and the corresponding labels in seperate arrays in the same order
known_face_encodings = [modi_face_encodings, trump_face_encodings, abhi_face_encodings, zeynep_face_encodings, serdar_face_encodings]
known_face_names = ["Narendra Modi", "Donald Trump", "Abhilash", "Zeynep Toker", "Serdar Koksal"]


#initialize the array variable to hold all face locations, encodings and names 
all_face_locations = []
all_face_encodings = []
all_face_names = []

#loop through every frame in the video
while True:
    #get the current frame from the video stream as an image
    ret,current_frame = webcam_video_stream.read()
    #resize the current frame to 1/4 size to proces faster
    current_frame_small = cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    #detect all faces in the image
    #arguments are image,no_of_times_to_upsample, model
    all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=1,model='hog')
    
    #detect face encodings for all the faces detected
    all_face_encodings = face_recognition.face_encodings(current_frame_small,all_face_locations)








    frame_weight = current_frame.shape[1]       #videodaki resim karelerinin genişliğini alır.
    frame_height = current_frame.shape[0]       #videodaki resim karelerinin boyunu alır.

    frame_blob = cv2.dnn.blobFromImage(current_frame, 1/255, (416,416), swapRB=True, crop=False) #resmin boyutları, resmi RGBye dönüştür, resmi kırpmaya gerek yok

    labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
              "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
              "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball",
              "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket",
              "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair",
              "sofa", "pottedplant", "bed", "diningtable", "toillet", "tvmonitor", "laptop", "mouse",
              "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator",
              "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"]


    colors = ["0,255,255", "0,0,255", "255,0,0", "255,255,0", "0,255,0"]
    colors = [np.array(color.split(",")).astype("int") for color in colors]
    colors = np.array(colors)
    colors = np.tile(colors, (18,1))


    model2 = cv2.dnn.readNetFromDarknet("C:/Users/Zeynep/Desktop/yolov3.cfg","C:/Users/Zeynep/Desktop/yolov3.weights")
    
    layers = model2.getLayerNames() #layers ile bütün katmanları çektik.
    output_layer = [layers[layer[0]-1] for layer in model2.getUnconnectedOutLayers()]
    
    model2.setInput(frame_blob)
    
    detection_layers = model2.forward(output_layer)

    ########## Non-Maximum Suppression - OPERATION 1 ##########
    
    ids_list = [] # id'lerin tutulduğu boş liste
    boxes_list = [] # boxes'ların tutulduğu boş liste
    confidences_list = [] # confidence'lerin tutulduğu boş liste
    
    ########## END OF OPERATION 1 ##########

    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            
            #detection değerlerinden en yüksek skorlu nesnemizi buluruz ->
            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]
            
            if confidence >0.35:
                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([frame_weight, frame_height, frame_weight, frame_height])
                (box_center_x, box_center_y, box_widht, box_height) = bounding_box.astype("int")
                
                start_x = int(box_center_x - (box_widht/2))
                start_y = int(box_center_y - (box_height/2))
    
                ########## Non-Maximum Suppression - OPERATION 2 ##########
                # yukarıdaki for döngüsünde tespit edilen her şey burada oluşturulan boş listelerin içerine yollandı
                ids_list.append(predicted_id)
                confidences_list.append(float(confidence))
                boxes_list.append([start_x, start_y,int(box_widht), int(box_height)])
    
                ################## END OF OPERATION 2 #####################
                
    ########## Non-Maximum Suppression - OPERATION 3 ##########
    
    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4) #en yüksek güvenirliğe sahip dikdörtgenleri gönderir.            
    
    for max_id in max_ids:
        max_class_id = max_id[0]
        box = boxes_list[max_class_id]
        
        start_x = box[0]
        start_y = box[1]
        box_widht = box[2]
        box_height = box[3]
    
        # boundig box'ların üzerine nesne ile ilgili label'ı yazabilmek için;
        predicted_id = ids_list[max_class_id] #max_class_id ile predectid_id ye eriştik
        label = labels[predicted_id] #predicted_id ile label'larımıza eriştik
        confidence = confidences_list[max_class_id] 
    
    ################## END OF OPERATION 3 #####################
    
        end_x = start_x + box_widht
        end_y = start_y + box_height
    
        box_color = colors[predicted_id]
        box_color = [int(each) for each in box_color]
                
        label = "{}: {:.2f}½".format(label, confidence*100)
        print("predicted object {}".format(label))
    
        cv2.rectangle(current_frame, (start_x, start_y), (end_x, end_y), box_color, 2)            
        cv2.putText(current_frame, label, (start_x, start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)









    #looping through the face locations and the face embeddings
    for current_face_location,current_face_encoding in zip(all_face_locations,all_face_encodings):
        #splitting the tuple to get the four position values of current face
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        
        #change the position maginitude to fit the actual size video frame
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        
        #find all the matches and get the list of matches
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding)
       
        #string to hold the label
        name_of_person = 'Unknown face'
        
        #check if the all_matches have at least one item
        #if yes, get the index number of face that is located in the first index of all_matches
        #get the name corresponding to the index number and save it in name_of_person
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]
        
        #draw rectangle around the face    
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(255,0,0),2)
        
        #display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pos,bottom_pos), font, 0.5, (255,255,255),1)
    
    #display the video
    cv2.imshow("Webcam Video",current_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release the stream and cam
#close all opencv windows open
webcam_video_stream.release()
cv2.destroyAllWindows()        
