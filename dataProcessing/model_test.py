import numpy as np
import tensorflow as tf
import h5py
import re
from tensorflow.keras.utils import Sequence
from sklearn.metrics import confusion_matrix, classification_report

class SEMGDataGenerator(Sequence):
    def __init__(self, hdf5_file, segments, batch_size=32, shuffle=False):
        self.hdf5_file = hdf5_file
        self.segments = segments  # List of segment names for the test set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()  # Shuffle the data if required
    
    def __len__(self):
        # Returns the number of batches per epoch
        return int(np.floor(len(self.segments) / self.batch_size))
    
    def __getitem__(self, index):
        # Generate one batch of data
        batch_segments = self.segments[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_segments)
        return X, y
    
    def on_epoch_end(self):
        # Optionally shuffle the data after each epoch
        if self.shuffle:
            np.random.shuffle(self.segments)
    
    def __data_generation(self, batch_segments):
        # Generate data for a batch
        X = []
        y = []
        
        with h5py.File(self.hdf5_file, 'r') as f:
            for segment_name in batch_segments:
                segment = f[segment_name]
                
                # Load and stack muscle data to form a 3D tensor
                muscles = list(segment.keys())
                muscles = [m for m in muscles if 'coeffs_normalized' in segment[m]]
                
                stacked_tensor = []
                for muscle in muscles:
                    coeffs = segment[muscle]['coeffs_normalized'][:]
                    stacked_tensor.append(coeffs)
                
                stacked_tensor = np.stack(stacked_tensor, axis=0)  # Stack along the new axis (muscles)
                X.append(stacked_tensor)
                
                # Automatically assign label based on the segment name (0 for Young, 1 for Old)
                if 'YOUNG' in segment_name.upper():
                    label = 0  # Young
                elif 'OLD' in segment_name.upper():
                    label = 1  # Old
                else:
                    raise ValueError(f"Unknown label in segment name: {segment_name}")
                
                y.append(label)
        
        # Convert lists to numpy arrays
        X = np.array(X)
        y = np.array(y)
        return X, y
    
# Function to extract subject ID and age group from the segment name
def extract_subject_info(segment_name):
    # Assuming subject ID is part of the segment name like 'IDxxxx'
    match = re.search(r'ID(\d+)', segment_name)  # Extracts subject ID
    subject_id = match.group(1) if match else None
    
    # Determine if the subject is young or old based on segment name
    age_group = 'YOUNG' if 'YOUNG' in segment_name.upper() else 'OLD' if 'OLD' in segment_name.upper() else None
    
    return subject_id, age_group

# Load data and trained model
hdf5_file = 'D:/sEMG_data/CWT_EMG_data_normalized.h5'
model = tf.keras.models.load_model('C:/Users/alexl/Desktop/dataProcessing/results_subjectDataSplit/final_model.h5')

# Load all segment names from the file
with h5py.File(hdf5_file, 'r') as f:
    all_segments = list(f.keys())
    
# Dictionary to store segments by subject
subject_dict = {}

# Group segments by subject
for segment_name in all_segments:
    subject_id, age_group = extract_subject_info(segment_name)
    
    if subject_id is not None and age_group is not None:
        if subject_id not in subject_dict:
            subject_dict[subject_id] = {'segments': [], 'age_group': age_group}
        subject_dict[subject_id]['segments'].append(segment_name)

# Separate subjects by age group
young_subjects = [subject_id for subject_id, info in subject_dict.items() if info['age_group'] == 'YOUNG']
old_subjects = [subject_id for subject_id, info in subject_dict.items() if info['age_group'] == 'OLD']

# Select 5 young and 5 old subjects for the test set
test_young_subjects = young_subjects[:5]
test_old_subjects = old_subjects[:5]

# Ensure the test subjects are only from these IDs
test_subjects = test_young_subjects + test_old_subjects

# Prepare test a data based on subject ID
test_segments = [segment_name for subject_id in test_subjects for segment_name in subject_dict[subject_id]['segments']]

# Create test data generators
test_data_generator = SEMGDataGenerator(hdf5_file, test_segments, batch_size=32, shuffle=False)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_data_generator)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# If you want to get the predicted classes and true labels from the test data:
y_pred = []
y_true = []

# Iterate through the test dataset to get predictions and true labels
for X_batch, y_batch in test_data_generator:
    preds = model.predict(X_batch)
    y_pred.extend(preds)
    y_true.extend(y_batch)

# Convert predictions to binary (assuming a binary classification problem with a sigmoid output)
y_pred = np.array(y_pred).flatten()
y_pred = (y_pred > 0.5).astype(int)

# Convert true labels to numpy array
y_true = np.array(y_true)

# Print the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Print a classification report (precision, recall, f1-score)
class_report = classification_report(y_true, y_pred)
print("Classification Report:")
print(class_report)