import csv
import os

image_names = os.listdir("/home/eslam/Masters/NN/IWild_project/YOLOv3_TensorFlow/data/my_data/training")
image_names = [x for x in image_names if '.jpg' in x]

data_labels = []
with open("/home/eslam/Masters/NN/IWild_project/YOLOv3_TensorFlow/data/my_data/training/train.csv",
          mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for element in csv_reader:
        if any(element['file_name'] in s for s in image_names):
            data_labels.append(
                {"file_name": element['file_name'], "category_id": element['category_id'], "height": element['height'],
                 "width": element['width']})
        print(element['file_name'])
        line_count += 1
    print("Finished", line_count)
