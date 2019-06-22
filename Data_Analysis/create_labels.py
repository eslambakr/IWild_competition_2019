import pickle

output_detection_path = "/media/eslam/D0FCBC10FCBBEF3A/iwild_data/detection_output/Detection_Results"
pickle_in = open(output_detection_path + "/CCT_Detection_Results_1.p", "rb")
example_dict = pickle.load(pickle_in)
print("Finish_Loading.....")
