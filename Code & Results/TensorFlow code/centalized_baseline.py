import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=全部,1=INFO,2=WARNING,3=ERROR
import tensorflow as tf
from keras.datasets import mnist

# Load the centralized version of the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

# Define the model (same as for the federated setting in EMNIST)
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax', kernel_initializer='zeros'),
])

# Compile the model with the same parameters
model.compile(
    optimizer=tf.keras.optimizers.SGD(0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# Train the model using the training dataset
model.fit(
    x_train, y_train,
    epochs=5
)

# Print the metrics of the model on the test dataset
test_metrics = model.evaluate(x_test, y_test)
print('test metrics={}'.format(test_metrics))

# Save the model
model.save('model_EMNIST_centralized.keras')
