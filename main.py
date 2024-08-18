import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Function to load images and labels from a folder
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (64, 64)) 
            images.append(img)
            if 'cat' in filename.lower():
                labels.append('cat')
            elif 'dog' in filename.lower():
                labels.append('dog')
    return images, labels

# Load cat and dog images
cat_folder = '/home/aditya2327/Documents/SVM/Images/cats'
dog_folder = '/home/aditya2327/Documents/SVM/Images/dogs'

cat_images, cat_labels = load_images_from_folder(cat_folder)
dog_images, dog_labels = load_images_from_folder(dog_folder)

# Combine images and labels
images = np.array(cat_images + dog_images)
labels = np.array(cat_labels + dog_labels)

# Encode labels to numeric values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Flatten the images for the SVM model
images = images.reshape(len(images), -1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create and train a Support Vector Machine model
svm = SVC(kernel='linear')  
svm.fit(X_train, y_train)

# Predict on the test set
y_pred = svm.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the trained model to a file
joblib.dump(svm, 'svm_cat_dog_classifier.pkl')

# Function to predict a new image
def predict_image(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img.flatten().reshape(1, -1)
    prediction = model.predict(img)
    return label_encoder.inverse_transform(prediction)

# Load the trained model from a file
svm = joblib.load('svm_cat_dog_classifier.pkl')

# Predict a new image
new_image_path = 'cat.4001.jpg'

prediction = predict_image(new_image_path, svm)
print(f'The predicted class is: {prediction[0]}')
