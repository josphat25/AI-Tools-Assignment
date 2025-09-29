# AI-Tools-Assignment
AI Tools Assignment: AI Tools and Applications  Objective &amp; Guidelines  This assignment evaluates your understanding of AI tools/frameworks and their real-world applications through a mix of theoretical and practical tasks. You’ll demonstrate proficiency in selecting, implementing, and critically analyzing AI tools to solve problems.  ##Part 1: Theoretical Understanding (40%)
Q1: Primary Differences Between TensorFlow and PyTorch
Aspect	TensorFlow	PyTorch
Computation Graph	Static (define-then-run), but TF 2.x introduced eager execution	Dynamic (define-by-run) – easier for debugging
Ease of Use	More boilerplate, but great for production and deployment (TF Serving, TF Lite)	More Pythonic and intuitive for research & prototyping
Ecosystem	Wide ecosystem: Keras (high-level API), TensorFlow Lite (mobile), TensorFlow.js (browser)	Focused on flexibility and research; production tools (TorchServe) are improving
Best Use Case	When deploying to production or working with large-scale models	When doing fast experimentation, research, and debugging

When to choose:

TensorFlow: Production-grade applications, mobile/edge deployment, scalability.

PyTorch: Research, prototyping, academic work, projects needing flexibility.

Q2: Two Use Cases for Jupyter Notebooks

Interactive Model Prototyping:
You can iteratively build and test ML models, visualize training curves, and adjust hyperparameters in real time.

Data Exploration and Visualization:
Allows you to load datasets, clean data, perform EDA (Exploratory Data Analysis), and visualize patterns before training models.

Q3: How spaCy Enhances NLP

Tokenization & NER: Instead of manually splitting strings with .split() or regex, spaCy provides robust tokenization, POS tagging, and Named Entity Recognition out of the box.

Pretrained Pipelines: spaCy uses optimized, pretrained statistical models that recognize entities like PERSON, ORG, PRODUCT, making NLP tasks more accurate and efficient.

Comparative Analysis: Scikit-learn vs TensorFlow
Criteria	Scikit-learn	TensorFlow
Target Applications	Classical ML (SVM, Decision Trees, Clustering, Regression)	Deep Learning (Neural Networks, CNNs, RNNs, Transformers)
Ease of Use	Easier for beginners – consistent API and less code	Slightly steeper learning curve, but Keras (inside TF) simplifies NN creation
Community Support	Strong for ML basics	Massive community, tutorials, and production-ready tools
Part 2: Practical Implementation (50%)
Task 1: Classical ML with Scikit-learn
# iris_classifier.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Handle missing values (if any)
df = df.dropna()

# Encode target labels
label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df['target'])

# Split data
X = df.drop(columns='target')
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))

Task 2: Deep Learning with TensorFlow
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add channel dimension
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Build CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)

# Visualize predictions
predictions = model.predict(x_test[:5])
for i in range(5):
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}, Actual: {y_test[i]}")
    plt.show()

Task 3: NLP with spaCy
import spacy
nlp = spacy.load("en_core_web_sm")

reviews = [
    "I love the new Apple iPhone 15, it's super fast!",
    "Samsung Galaxy is a great phone but a bit pricey."
]

for review in reviews:
    doc = nlp(review)
    print(f"\nReview: {review}")
    print("Entities:")
    for ent in doc.ents:
        print(f"  {ent.text} ({ent.label_})")
    sentiment = "Positive" if any(word in review.lower() for word in ["love","great","awesome"]) else "Negative"
    print(f"Sentiment: {sentiment}")

Part 3: Ethics & Optimization (10%)
Ethical Considerations

Potential Bias:

MNIST is grayscale, so models may not generalize well to colored digits.

Amazon reviews may be biased by demographics or fake reviews.

Mitigation:

Use tools like TensorFlow Fairness Indicators to check performance across subgroups.

Augment training data to balance classes and improve fairness.

Use spaCy’s rule-based systems to filter offensive language or detect biased terms.

Troubleshooting Challenge

Typical Issues to Fix:

Dimension mismatch: Ensure input shape matches (28,28,1).

Wrong loss function: Use sparse_categorical_crossentropy for integer labels.

Incorrect output layer: Use softmax with Dense(10) for 10 classes.

Bonus Task (Extra 10%) – Deployment
Using Streamlit
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("mnist_cnn.h5")
st.title("MNIST Digit Classifier")

uploaded_file = st.file_uploader("Upload a digit image (28x28)", type=["png","jpg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("L").resize((28,28))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, (0,-1))
    prediction = model.predict(img_array)
    st.image(img, caption=f"Predicted: {np.argmax(prediction)}", width=150)


Run locally:

streamlit run app.py
