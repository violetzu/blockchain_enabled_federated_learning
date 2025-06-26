import collections
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=全部,1=INFO,2=WARNING,3=ERROR
import tensorflow as tf
import tensorflow_federated as tff
import random
import time
from datetime import datetime
from matplotlib import pyplot as plt
from monitor import TrainingMonitor

np.random.seed(1000)

tf.compat.v1.enable_v2_behavior()
tf.compat.v1.graph_util.extract_sub_graph

# 0. PARAMETERS

# NUM_CLIENTS = 200
# 用於集中式評估時，隨機抽取多少個 client 的資料作為測試集
NUM_CLIENTS_TEST = 50
# 每個 client 在本地訓練時，完整遍歷自己資料的次數
NUM_EPOCHS = 5
# 在本地訓練和評估時，每個 batch 包含多少筆樣本
BATCH_SIZE = 20
# 在做 shuffle() 時，維持的 buffer 大小。
# buffer 越大，打亂程度越好，但同時也佔用更多記憶體
SHUFFLE_BUFFER = 100
# 在資料 pipeline 裡，預先抓取多少個 batch 到記憶體
# 這能提高資料讀取和運算的併發，但也會佔用額外 RAM
PREFETCH_BUFFER = 10
# 聯邦學習整個過程要跑多少個通訊輪次（rounds）
NUM_ROUNDS_FL = 200

#變動參數
NUM_CLASSES_PER_USER = 3 #每個 client dataset保留的樣本類別數
PARTITIONS = [100, 200] #client數量
PERCENTAGES = [0.1, 0.25, 0.5, 0.75, 1] #參與比例
# PERCENTAGES = [1]
# 1. METHODS


# Pre-processing function
def preprocess(dataset):
    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return collections.OrderedDict(
            x=tf.reshape(element['pixels'], [-1, 784]),
            y=tf.reshape(element['label'], [-1, 1]))
    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(
        BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


# Make data federated
def make_federated_data(client_data, client_ids):
    return [
        preprocess(client_data.create_tf_dataset_for_client(x))
        for x in client_ids
    ]


def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(784)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax(),
    ])


# 2. LOAD DATA

# Load the mnist dataset
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()


client_ids_list = []
client_ids_list_ix = []

# Get the clients IDs in str and int forms
for i in range(0, len(emnist_train.client_ids)-1):
    client_ids_list.append(emnist_train.client_ids[i])
    client_ids_list_ix.append(i)

# 先隨機抽 client 專屬的 class set，放進全域 dict，避免每次都重新抽
CLIENT_ALLOWED_CLASSES = {
    i: np.random.choice(10, NUM_CLASSES_PER_USER, replace=False).astype(np.int32)
    for i in client_ids_list_ix
}

def create_tf_dataset_for_client_fn(client_id):
    allowed = tf.constant(CLIENT_ALLOWED_CLASSES[client_id])
    ds = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[client_id])

    # tf.data 完整在圖裡運算 → 不會用到 PyFunc
    def _keep(example):
        return tf.reduce_any(tf.equal(example['label'], allowed))

    return ds.filter(_keep)


# Generate the new training dataset
pruned_emnist_train = tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
    client_ids_list_ix, create_tf_dataset_for_client_fn)

# Define the ML model compliant with FL (needs a sample of the dataset to be defined)
sample_dataset = pruned_emnist_train.create_tf_dataset_for_client(pruned_emnist_train.client_ids[0])
preprocessed_sample = preprocess(sample_dataset)


f = plt.figure(figsize=(12,7))
f.suptitle("Label Counts for a sample of clients")
for i in range(6):
    # print(pruned_emnist_train.client_ids[i])
    client_dataset = pruned_emnist_train.create_tf_dataset_for_client(pruned_emnist_train.client_ids[i])
    plot_data = collections.defaultdict(list)
    for example in client_dataset:
        label = example['label'].numpy()
        plot_data[label].append(label)
    plt.subplot(2,3,i+1)
    plt.title("Client {}".format(i))
    for j in range(10):
        plt.hist(plot_data[j], density = False, bins = [0,1,2,3,4,5,6,7,8,9,10])
plt.savefig("./test_figure.jpg")

len_client_dataset = [0] * len(emnist_train.client_ids)
len_client_dataset_test = [0] * len(emnist_train.client_ids)
for i in range(0, len(emnist_train.client_ids)-1):
    client_dataset = pruned_emnist_train.create_tf_dataset_for_client(pruned_emnist_train.client_ids[i])
    for example in client_dataset:
        len_client_dataset[i] += 1
    client_dataset_test = emnist_test.create_tf_dataset_for_client(emnist_test.client_ids[i])
    for example in client_dataset_test:
        len_client_dataset_test[i] += 1

print('TRAIN')
print(len(len_client_dataset))
print(sum(len_client_dataset))
print(np.mean(len_client_dataset))

print('TEST')
print(len(len_client_dataset_test))
print(sum(len_client_dataset_test))
print(np.mean(len_client_dataset_test))


def model_fn():  # Model constructor (needed to be passed to TFF, instead of a model instance)
    keras_model = create_keras_model()
    # 新版 API：從 tff.learning.models 取
    return tff.learning.models.from_keras_model(
        keras_model=keras_model,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        input_spec=preprocessed_sample.element_spec,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

fed_evaluation = tff.learning.algorithms.build_fed_eval(model_fn)

# Define the iterative process to be followed for training both clients and the server
# More optimizers here: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
LEARNING_RATE_CLIENT = 0.01
LEARNING_RATE_SERVER = 1.00
iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn=model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_CLIENT),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE_SERVER))

# 3. TRAIN A MODEL
eval_state = fed_evaluation.initialize()
for partition in PARTITIONS:

    print(' + Dataset size: ' + str(partition))
    # Take a subset of the dataset
    subset_ix = np.random.choice(client_ids_list_ix, partition)
    # print(subset_ix)

    for percentage in PERCENTAGES:

        print(f'     -{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} START Training percentage:{percentage}')
        monitor = TrainingMonitor()

        train_loss = []
        train_accuracy = []
        test_loss = []
        test_accuracy = []
        eval_loss = []
        eval_accuracy = []
        training_time = []
        iteration_time = []

        # Initialize the server state
        state = iterative_process.initialize()
        # Iterate for each communication round in FL
        for round_num in range(0, NUM_ROUNDS_FL):

            round_time_start = time.time()

            # Training round
            #  - Get a random sample of clients
            # participating_clients = np.random.choice(emnist_train.client_ids, size=NUM_CLIENTS)  # emnist_train.client_ids[0:NUM_CLIENTS-1]
            training_ixs = np.random.choice(subset_ix, round(partition*percentage), replace=False) #emnist_train.client_ids[0:round(partition*percentage)]

            train_datasets = make_federated_data(pruned_emnist_train, training_ixs)
            train_time_start = time.time()
            state, train_metrics = iterative_process.next(state, train_datasets)
            # print('  - Round  {}, train metrics={}'.format(round_num, train_metrics))
            train_time_end = time.time()
            training_time.append(train_time_end - train_time_start)
            #  - Get training metrics
            train_loss.append(train_metrics['client_work']['train']['loss'])
            train_accuracy.append(train_metrics['client_work']['train']['sparse_categorical_accuracy'])
            #  - Get test metrics
            ixes_original = []
            for i in training_ixs:
                ixes_original.append(emnist_test.client_ids[i])
            test_datasets = make_federated_data(emnist_test, ixes_original)
            weights = iterative_process.get_model_weights(state)
            eval_state = fed_evaluation.set_model_weights(eval_state, weights)
            evaluation_output = fed_evaluation.next(eval_state, test_datasets)
            test_metrics = evaluation_output.metrics
            eval_state = evaluation_output.state   # 保留最新狀態
            test_loss.append(test_metrics['client_work']['eval']['current_round_metrics']['loss'])
            test_accuracy.append(test_metrics['client_work']['eval']['current_round_metrics']['sparse_categorical_accuracy'])

            # Choose another set of random clients for evaluation
            sample_random_clients_ids = random.sample(range(0, len(emnist_test.client_ids) - 1), NUM_CLIENTS_TEST)
            sample_random_clients = []
            for idx in sample_random_clients_ids:
                sample_random_clients.append(emnist_test.client_ids[idx])
            eval_datasets = make_federated_data(emnist_test, sample_random_clients)
            evaluation_output = fed_evaluation.next(eval_state, eval_datasets)
            eval_metrics = evaluation_output.metrics
            eval_state = evaluation_output.state
            eval_loss.append(eval_metrics['client_work']['eval']['current_round_metrics']['loss'])
            eval_accuracy.append(eval_metrics['client_work']['eval']['current_round_metrics']['sparse_categorical_accuracy'])

            round_time_end = time.time()
            iteration_time.append(round_time_end - round_time_start)

        monitor.info()
        # SAVE RESULTS
        output_dir = f'OUTPUTS/sync_vs_async/num_classes_{NUM_CLASSES_PER_USER}/'
        os.makedirs(output_dir, exist_ok=True)
        np.savetxt(f'{output_dir}train_loss_K{partition}_{percentage}.txt',     np.reshape(train_loss, (1, NUM_ROUNDS_FL)))
        np.savetxt(f'{output_dir}train_accuracy_K{partition}_{percentage}.txt', np.reshape(train_accuracy, (1, NUM_ROUNDS_FL)))
        np.savetxt(f'{output_dir}test_loss_K{partition}_{percentage}.txt',      np.reshape(test_loss, (1, NUM_ROUNDS_FL)))
        np.savetxt(f'{output_dir}test_accuracy_K{partition}_{percentage}.txt',  np.reshape(test_accuracy, (1, NUM_ROUNDS_FL)))
        np.savetxt(f'{output_dir}eval_loss_K{partition}_{percentage}.txt',      np.reshape(eval_loss, (1, NUM_ROUNDS_FL)))
        np.savetxt(f'{output_dir}eval_accuracy_K{partition}_{percentage}.txt',  np.reshape(eval_accuracy, (1, NUM_ROUNDS_FL)))
        np.savetxt(f'{output_dir}iteration_time_K{partition}_{percentage}.txt', np.reshape(iteration_time, (1, NUM_ROUNDS_FL)))