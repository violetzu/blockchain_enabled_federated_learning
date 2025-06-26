import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=全部,1=INFO,2=WARNING,3=ERROR
import tensorflow as tf
from keras.datasets import mnist
import nest_asyncio

nest_asyncio.apply()

tf.compat.v1.enable_v2_behavior()
tf.compat.v1.graph_util.extract_sub_graph

# Final evaluation in test dataset
print("Evaluating the entire test dataset:")
# Load the model
final_model = tf.keras.models.load_model('./model_EMNIST_centralized.h5')
print(final_model.get_weights())
# Load the centralized version of the EMNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"Evaluating the model in {x_test.shape[0]} test samples")
# Evaluate the final model on the test dataset
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
final_eval = final_model.evaluate(x_test, y_test)
print(final_eval)
print('Test loss:', final_eval[0])
print('Test accuracy:', final_eval[1])

# SAVE THE RESULTS
# # Final accuracy on the entire test dataset
# np.savetxt('final_test_loss_K' + str(m) + '_' + str(percentage) + '.txt', np.reshape(final_eval[0], (1,1)))
# np.savetxt('final_test_accuracy_K' + str(m) + '_' + str(percentage) + '.txt', np.reshape(final_eval[1], (1,1)))

