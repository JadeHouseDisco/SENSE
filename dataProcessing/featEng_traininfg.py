import tensorflow as tf
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense, Concatenate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

class SEMGDataGenerator(Sequence):
    def __init__(self, hdf5_file, segment_names, batch_size=4, shuffle=True):
        self.hdf5_file = hdf5_file
        self.segments = segment_names
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()  # Shuffle the data if required
    
    def __len__(self):
        # Returns the number of batches per epoch
        return int(np.floor(len(self.segments) / self.batch_size))
    
    def __getitem__(self, index):
        # Generate one batch of data
        batch_segments = self.segments[index * self.batch_size:(index + 1) * self.batch_size]
        X, y, gender_features = self.__data_generation(batch_segments)
        return [X, gender_features], y  # Return data and gender feature as separate inputs
    
    def on_epoch_end(self):
        # Optionally shuffle the data after each epoch
        if self.shuffle:
            np.random.shuffle(self.segments)
    
    def __data_generation(self, batch_segments):
        X = []
        y = []
        gender_features = []

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
                
                stacked_tensor = np.stack(stacked_tensor, axis=-1)  # Stack along the new axis (muscles)
                X.append(stacked_tensor)
                
                # Label based on the segment name (0 for Young, 1 for Old)
                if 'YOUNG' in segment_name.upper():
                    label = 0  # Young
                elif 'OLD' in segment_name.upper():
                    label = 1  # Old
                else:
                    raise ValueError(f"Unknown label in segment name: {segment_name}")
                
                # Extract gender information from the segment name (1 for male, 0 for female)
                if '_M_' in segment_name:
                    gender = 1  # Male
                elif '_F_' in segment_name:
                    gender = 0  # Female
                else:
                    raise ValueError(f"Unknown gender in segment name: {segment_name}")
                
                y.append(label)
                gender_features.append(gender)

        # Convert lists to numpy arrays
        X = np.array(X)
        y = np.array(y)
        gender_features = np.array(gender_features).reshape(-1, 1)  # Reshape to (batch_size, 1)
        
        return X, y, gender_features
    
# Function to extract the dimensions of the data (number_of_muscles, frequency_bins, time_points)
def extract_data_dimensions(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        # Get the first segment
        first_segment = list(f.keys())[0]
        segment = f[first_segment]
        
        # Assuming all segments have consistent size, extract one muscle group
        first_muscle = list(segment.keys())[0]
        muscle_data = segment[first_muscle]['coeffs_normalized'][:]
        
        # The shape of the muscle data will give us frequency_bins (rows) and time_points (columns)
        frequency_bins, time_points = muscle_data.shape
        
        # The number of muscles is the number of keys in the segment
        number_of_muscles = len([m for m in segment.keys() if 'coeffs_normalized' in segment[m]])
    
    return number_of_muscles, frequency_bins, time_points

# Function to extract subject ID and age group from the segment name
def extract_subject_info(segment_name):
    # Assuming subject ID is part of the segment name like 'IDxxxx'
    match = re.search(r'ID(\d+)', segment_name)  # Extracts subject ID
    subject_id = match.group(1) if match else None
    
    # Determine if the subject is young or old based on segment name
    age_group = 'YOUNG' if 'YOUNG' in segment_name.upper() else 'OLD' if 'OLD' in segment_name.upper() else None
    
    return subject_id, age_group

if __name__ == "__main__":
    hdf5_file = 'D:/sEMG_data/CWT_EMG_data_normalized.h5'
    number_of_muscles, frequency_bins, time_points = extract_data_dimensions(hdf5_file) # 13, 10000, 47

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

    # Prepare test and training data based on subject ID
    test_segments = [segment_name for subject_id in test_subjects for segment_name in subject_dict[subject_id]['segments']]
    train_segments = [segment_name for subject_id, info in subject_dict.items() if subject_id not in test_subjects for segment_name in info['segments']]

    # Create training and test data generators
    train_data_generator = SEMGDataGenerator(hdf5_file, train_segments, batch_size=16)
    test_data_generator = SEMGDataGenerator(hdf5_file, test_segments, batch_size=16, shuffle=False)

   # CNN model for sEMG data
    sEMG_input = Input(shape=(frequency_bins, time_points, number_of_muscles), name='sEMG_input')
    x = Conv2D(32, (3, 3), activation='relu')(sEMG_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    # Dense layers for gender input
    gender_input = Input(shape=(1,), name='gender_input')  # Input for gender feature

    # Concatenate CNN and gender feature
    concatenated = Concatenate()([x, gender_input])

    # Final dense layers
    x = Dense(64, activation='relu')(concatenated)
    output = Dense(1, activation='sigmoid')(x)

    # Define model
    model = tf.keras.models.Model(inputs=[sEMG_input, gender_input], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('C:/Users/alexl/Desktop/dataProcessing/best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    # Train the model using the training data generator
    history = model.fit(
        train_data_generator, 
        epochs=50, 
        validation_data=test_data_generator, 
        callbacks=[early_stopping, model_checkpoint, reduce_lr], 
        workers=4, 
        use_multiprocessing=False
    )


    # Train the model using the training data generator
    history = model.fit(
        train_data_generator, 
        epochs=50, 
        validation_data=test_data_generator, 
        callbacks=[early_stopping, model_checkpoint, reduce_lr], 
        workers=4, 
        use_multiprocessing=False
    )

    model.save('C:/Users/alexl/Desktop/dataProcessing/final_model.h5')

    # Evaluate the model on the test data generator
    test_loss, test_accuracy = model.evaluate(test_data_generator, workers=4, use_multiprocessing=True)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # Visualize training results
    # Plot accuracy over epochs
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot loss over epochs
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()