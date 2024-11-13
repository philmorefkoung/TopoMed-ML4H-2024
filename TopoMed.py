import pandas as pd
import numpy as np
import random
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split

os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Step 1: Set Seed, Num Classes, Input Dimension, Batch Size
set_seed(9)
n_classes = 3
input_dim = 400
batch_size = 128

# Step 2: Load and pre-process datasets
train = pd.read_csv("\Path\to\Training\Dataset.csv")
val = pd.read_csv("\Path\to\Validation\Dataset.csv")
test = pd.read_csv("\Path\to\Testing\Dataset.csv")

train_label = train[['Label']].copy()
val_label = val[['Label']].copy()
test_label = test[['Label']].copy()

train = train.drop(columns=['Label'])
val = val.drop(columns=['Label'])
test= test.drop(columns=['Label'])

# Transform Labels
label_encoder = LabelEncoder() 

train_label = label_encoder.fit_transform(train_label) 
train_label = tf.keras.utils.to_categorical(train_label, n_classes) 

val_label = label_encoder.transform(val_label) 
val_label = tf.keras.utils.to_categorical(val_label, n_classes) 

test_label = label_encoder.transform(test_label) 
test_label = tf.keras.utils.to_categorical(test_label, n_classes) 

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train, train_label))
val_dataset = tf.data.Dataset.from_tensor_slices(((val), val_label))
test_dataset = tf.data.Dataset.from_tensor_slices((test, test_label))

# Shuffle and batch the training dataset and batch val and test datasets
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Step 3: Build Model
def lr_schedule(epoch):
    lr = 0.001
    if epoch < 50:
        return lr
    elif epoch < 75:
        return lr * 0.01
    else:
        return lr * 0.01 * 0.01  
    
lr_scheduler = LearningRateScheduler(lr_schedule)

model = Sequential([
    Input(shape=(input_dim,)),          
    Dense(396, activation='relu'),
    Dropout(0.1),
    Dense(198, activation='relu'),
    Dropout(0.1),
    Dense(99, activation='relu'),
    Dropout(0.1),
    Dense(n_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', AUC(name='AUC'), Precision(name='Precision'), Recall(name='Recall')])

# Step 4: Train Model
model.fit(train_dataset, 
              epochs=100,
              validation_data=val_dataset,
              callbacks=[lr_scheduler])

# Step 5: Evaluate Model
test_loss, test_acc, test_auc, test_precision, test_recall= model.evaluate(test_dataset)

print(f'Test accuracy: {test_acc}')
print(f'Test AUC: {test_auc}')
print(f'Test Precision: {test_precision}')
print(f'Test Recall: {test_recall}')