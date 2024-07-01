from ultralytics import YOLO

# Loading the YOLOv8 (Version 8) Model
model = YOLO('yolov8n.pt')

# Running inference on the source
'''in the source argument:
- give the path of the test file that you want to run the model on, if the test file is in a different folder
- give the name of the file if the the file is in the present owrking directory only
- give source = 0 if using the default webcam for live streaming'''
results = model(source='name_of_test_file', show=True, conf=0.4, save=True)
