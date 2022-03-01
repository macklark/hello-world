from tensorflow import keras
from keras.datasets import mnist
from keras import layers
import mlflow
import sys

# mlflow.set_tracking_uri("http://ec2-3-145-88-77.us-east-2.compute.amazonaws.com:5000/")

# mlflow.set_tracking_uri("https://dagshub.com/macklark/mlflow-tracking.mlflow")

# mlflow.set_tracking_uri("http://127.0.0.1:5000/")
# mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_tracking_uri("http://ec2-3-138-116-153.us-east-2.compute.amazonaws.com:5000/")

epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])
experimentId = float(sys.argv[3])

experiment_id = mlflow.create_experiment('mnist_experiment_{}'.format(experimentId))
# experiment_id = mlflow.create_experiment('mnist_experiment_0.2')

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

with mlflow.start_run(experiment_id=experiment_id):
    model = keras.Sequential([
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)

    train_loss, train_accu = model.evaluate(train_images, train_labels)
    test_loss, test_accu = model.evaluate(test_images, test_labels)

    mlflow.log_param("epochs", 6)
    mlflow.log_param("batch_size", 128)

    mlflow.log_metric("accuracy_on_training", train_accu)
    mlflow.log_metric("accuracy_on_testing", test_accu)

    mlflow.keras.log_model(model, "keras_model")

