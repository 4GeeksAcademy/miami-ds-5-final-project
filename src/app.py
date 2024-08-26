import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from sklearn.cluster import KMeans
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.callbacks import TensorBoard
from keras import Sequential
from keras.applications.vgg16 import preprocess_input
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization
from keras.utils import to_categorical
import tensorflow as tf
import warnings
import datetime
import os


def warn(*args, **kwargs):
    pass


class LRSched(tf.keras.callbacks.Callback):
    def __init__(self, models):
        super().__init__()
        self.models = models
        self.base_lrs = {i: model.optimizer.learning_rate.numpy() for i, model in models.items()}
        self.previous_losses = {i: float('inf') for i in models.keys()}

    def on_epoch_end(self, epoch, logs=None):
        for i, model in self.models.items():
            current_lr = model.optimizer.learning_rate.numpy()
            val_loss = logs.get(f'val_loss_{i}', None)
            if val_loss is not None:
                if val_loss < self.previous_losses[i]:
                    new_lr = current_lr * 0.9
                else:
                    new_lr = min(current_lr * 1.1, self.base_lrs[i])
                model.optimizer.learning_rate.assign(new_lr)
                self.previous_losses[i] = val_loss


class FFTT:
    with open('data.dat', 'rb') as file:
        train_df, test_df = pickle.load(file)
    train_df['label'] = [i[0] for i in train_df['label']]
    train_df['label'] = to_categorical(train_df['label'], num_classes=9)
    train_df = train_df[['img', 'label']]
    test_df['label'] = [i[0] for i in test_df['label']]
    test_df['label'] = to_categorical(test_df['label'], num_classes=9)
    test_df = test_df[['img', 'label']]
    with open('weights.dat', 'rb') as file:
        weights = pickle.load(file)
    with open('kmodel.dat', 'rb') as file:
        kmodel = pickle.load(file)

    def __init__(self):
        try:
            with open('models.dat', 'rb') as file:
                self.models = pickle.load(file)
        except FileNotFoundError:
            self.models = {i: Sequential() for i in range(4)}
            self.model_init()
            # self.preweight()
        self.lrs = LRSched(self.models)
        self.model_skip = []
        self.prepare()
        self.epochs = 20
        self.batch_size = 250
        self.patience = 6
        self.best_acc = 0
        self.loss_stop = []
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        self.stopped = False
        self.train()

    def model_init(self):
        for i in self.models.values():
            i.add(
                Conv2D(input_shape=(28, 28, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
            i.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
            i.add(BatchNormalization())
            i.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
            i.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
            i.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
            i.add(BatchNormalization())
            i.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
            i.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
            i.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
            i.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
            i.add(BatchNormalization())
            i.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
            i.add(Flatten())
            i.add(Dense(units=4096, activation="relu"))
            i.add(Dense(units=4096, activation="relu"))
            i.add(Dense(units=9, activation="softmax"))
            i.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
                      metrics=['accuracy'])

    def preweight(self):
        for i in self.weights.keys():
            if i == 1:
                original_weights, original_bias = self.models[i].layers[-1].get_weights()
                if len(self.weights[i]) != 9:
                    self.weights[i].insert(0, 0)
                    self.weights[i].insert(3, 0)
                    self.weights[i].insert(6, 0)
                mod_weights = original_weights * np.array(self.weights[i]).reshape(1, 9)
                self.models[i].layers[-1].set_weights([mod_weights, original_bias])

    def preprocess(self, img):
        red, green, blue = img[:, :, 0].flatten().tolist(), img[:, :, 1].flatten().tolist(), img[:, :,
                                                                                             2].flatten().tolist()
        colors = {'red': red, 'green': green, 'blue': blue}
        funcs = {'_avg': np.mean, '_std': np.std, '_max': np.max, '_min': np.min}
        results = {}
        for _name, func in funcs.items():
            for name, color in colors.items():
                results[name + _name] = func(color)
        model_num = self.kmodel.predict(pd.DataFrame(results, index=[0]))[0]
        img = img / 255.0
        img = img.astype('float32')
        return img, model_num

    def prepare(self):
        img = self.train_df['img'].apply(self.preprocess)
        self.train_df['img'] = [i[0] for i in img]
        self.train_df['group'] = [i[1] for i in img]
        img = self.test_df['img'].apply(self.preprocess)
        self.test_df['img'] = [i[0] for i in img]
        self.test_df['group'] = [i[1] for i in img]

    def batch(self, batch_size, drop_last=True, test=False):
        with tf.device('/GPU:0'):
            df = self.train_df if not test else self.test_df

            def generator():
                for img, label, group in zip(df['img'].values, df['label'].values, df['group'].values):
                    img = tf.convert_to_tensor(img.astype(np.float32))
                    label = tf.convert_to_tensor(label, dtype=tf.float32)
                    label = to_categorical(label, num_classes=9)
                    group = tf.convert_to_tensor(group, dtype=tf.int64)
                    yield img, label, group

            dataset = tf.data.Dataset.from_generator(
                generator,
                output_signature=(
                    tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(9,), dtype=tf.float32),  # Assuming label has 9 classes
                    tf.TensorSpec(shape=(), dtype=tf.int64)
                )
            )

            def map_fn(img, label, group):
                return img, (label, group)

            dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(batch_size, drop_remainder=drop_last).prefetch(tf.data.AUTOTUNE)

            return dataset

    # def batch(self, batch_size, drop_last=True, test=False):
    #     if not test:
    #         df = self.train_df
    #     else:
    #         df = self.test_df
    #     while True:
    #         for start in range(0, len(df), batch_size):
    #             if len(df) - start < batch_size:
    #                 if drop_last:
    #                     False
    #             end = min(start + batch_size, len(df))
    #             batch = df[start:end]
    #             imgs = [img for img in batch['img']]
    #             groups = [group for group in batch['group']]
    #             labels = np.array(batch['label'])
    #             labels = to_categorical(labels, num_classes=9)
    #             model_breakdown = {i:[[],[]] for i in range(4)}
    #             for i in range(len(imgs)):
    #                 model_breakdown[groups[i]][0].append(imgs[i])
    #                 model_breakdown[groups[i]][1].append(labels[i])
    #             for i in model_breakdown.keys():
    #                 model_breakdown[i] = [np.array(model_breakdown[i][0]), np.array(model_breakdown[i][1])]
    #             yield model_breakdown

    def train(self):
        tf.profiler.experimental.start(logdir='logdir')
        with tf.device('/GPU:0'):
            save_count = 0
            for epoch in range(self.epochs):
                if len(self.model_skip) == len(self.models):
                    with open('models.dat', 'wb') as file:
                        pickle.dump(self.models, file)
                    break
                print(f'Epoch {epoch + 1}/{self.epochs}')
                count = 0
                self.losses = {i: [] for i in range(4)}
                self.accuracies = {i: [] for i in range(4)}
                for images, (labels, model_idx) in tqdm(self.batch(self.batch_size),
                                                        total=len(self.train_df) // self.batch_size,
                                                        desc=f'Training Epoch {epoch + 1}'):
                    count += 1
                    tf.keras.backend.clear_session()
                    model_idx = model_idx.numpy()[0]
                    if model_idx in self.model_skip:
                        continue
                    if len(images) > 0:
                        with tf.profiler.experimental.Trace('train_step', step_num=count, _r=1):
                            loss, acc = self.models[model_idx].train_on_batch(images, labels)
                            self.losses[model_idx].append(loss)
                            self.accuracies[model_idx].append(acc)
                    if count == 300:
                        if not self.stopped:
                            tf.profiler.experimental.stop()
                            self.stopped = True
                for i in self.losses.keys():
                    if len(self.losses[i]) != 0:
                        print(
                            f'Model {i}: \nLoss: {sum(self.losses[i]) / len(self.losses[i])}\nAccuracy: {sum(self.accuracies[i]) / len(self.accuracies[i])}')
                total_test_loss = 0
                total_test_acc = 0
                num_batches = 0
                self.losses = {i: [] for i in range(4)}
                self.accuracies = {i: [] for i in range(4)}
                val_losses = {i: [] for i in range(4)}
                for images, (labels, model_idx) in tqdm(self.batch(self.batch_size, test=True),
                                                        total=len(self.test_df) // self.batch_size,
                                                        desc=f'Validation Epoch {epoch + 1}'):
                    tf.keras.backend.clear_session()
                    if len(images) > 0:
                        model_idx = model_idx.numpy()[0]
                        preds = self.models[model_idx].predict_on_batch(images)
                        val_loss = self.models[model_idx].evaluate(images, labels, verbose=0)[0]
                        self.losses[model_idx].append(val_loss)
                        val_losses[model_idx].append(val_loss)
                        total_test_loss += val_loss
                        preds = np.argmax(preds, axis=-1)
                        labels = np.argmax(labels, axis=-1)
                        acc = np.mean(preds == labels)
                        self.accuracies[model_idx].append(acc)
                        total_test_acc += acc
                        num_batches += 1
                logs = {f'val_loss_{model_idx}': sum(val_losses[model_idx]) / len(val_losses[model_idx]) for model_idx
                        in self.models.keys()}
                self.lrs.on_epoch_end(epoch, logs)
                avg_test_loss = total_test_loss / num_batches
                avg_test_acc = total_test_acc / num_batches
                print(f'Test Loss: {avg_test_loss}\nTest Accuracy: {avg_test_acc}')
                for i in self.losses.keys():
                    if len(self.losses[i]) != 0:
                        if sum(self.losses[i]) / len(self.losses[i]) < .9 < sum(self.accuracies[i]) / len(
                                self.accuracies[i]) and i not in self.model_skip:
                            self.model_skip.append(i)
                        print(
                            f'Model {i}:\nLoss: {sum(self.losses[i]) / len(self.losses[i])}\nAccuracy: {sum(self.accuracies[i]) / len(self.accuracies[i])}')
                if avg_test_acc > self.best_acc:
                    with open(f'{save_count}models.dat', 'wb') as file:
                        pickle.dump(self.models, file)
                    save_count += 1
                # if self.loss_stop == []:
                #     self.loss_stop = [avg_test_loss, self.patience]
                # else:
                #     if avg_test_loss >= self.loss_stop[0]:
                #         self.loss_stop[1] -= 1
                #     else:
                #         self.loss_stop = [avg_test_loss, self.patience]
                # if self.loss_stop[1] == 0:
                #     print(f'Early Stopping on Epoch {epoch + 1}\nLoss plateaued at {self.loss_stop[0]}')
                #     break


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses TensorFlow logs
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/dev/null'
    os.environ['XLA_PTXAS_CONFIG'] = '--Wno-deprecated-gpu-targets'
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
        except RuntimeError as e:
            print(e)

    warnings.warn = warn
    FFTT()
