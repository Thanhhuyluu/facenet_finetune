# Face Recognition

USE YOUR OWN DATASET

- Step 1: Copy your dataset and paste into project folder
- Step 2: Align images in dataset with mtcnn: enter python align_dataset_mtccn.py <input_folder> <output_folder>
- Step 3: Train using online_triplet_mining_train.py(choosing triplets base on current weights while training) or train_triplet.py(this will train choosing triplets randomly) 

HOW TO USE APP
- Step 1: Align images of people need to be considered "Verified" and paste in to ./database
- Step 2: Run App.py

*This project was inspired and base on this post on Medium: https://medium.com/@mohitsaini_54300/train-facenet-with-triplet-loss-for-real-time-face-recognition-a39e2f4472c3