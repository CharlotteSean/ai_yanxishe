# coding=utf-8
# author=yphacker

import os
import re
import soundfile
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
from audio_processor import AudioProcessor
from pydub import AudioSegment
from conf import config
from utils.utils import prepare_model_settings, load_label_dict

audio_processor = AudioProcessor()
model_settings = prepare_model_settings()
audio_processor.prepare_processing_graph(model_settings)
label_dict, label_dict_res = load_label_dict()


def deal_x(item):
    path = str(item['audio_and_txt_files_path'])
    wav_path = os.listdir(config.data_path + '/' + path)
    wav_list = []
    sounds = []
    for wavs in wav_path:
        wav = path + '/'
        if re.match('.*.wav', wavs):
            data, samplerate = soundfile.read(config.data_path + '/' + path + '/' + wavs)
            soundfile.write(config.data_path + '/' + path + '/' + wavs, data, samplerate, subtype='PCM_16')
            sounds.append(AudioSegment.from_wav(config.data_path + '/' + path + '/' + wavs))
            wav += wavs
            wav_list.append(wav)
    wav = wav_list[0]
    with tf.Session() as sess:
        data = audio_processor.get_data(os.path.join(config.data_path, wav), model_settings, 0, sess)
    return np.squeeze(data, axis=0)


def deal_y(diagnosis):
    y = np.zeros((config.num_labels,))
    y[label_dict[diagnosis]] = 1
    return y


def train():
    # df = pd.read_csv(config.train_path)
    # df = df.sample(frac=1).reset_index(drop=True)
    # x_train = df.apply(lambda row: deal_x(row), axis=1)
    # y_train = df['diagnosis'].apply(lambda x: deal_y(x))
    # # np.save('../data/x_train.npy', x_train)
    # # np.save('../data/y_train.npy', y_train)
    # df['x_train'] = x_train
    # df['y_train'] = y_train
    # df = pd.concat([df, df[df['diagnosis'] != 'COPD']])
    # df = df.sample(frac=1).reset_index(drop=True)
    # np.save('../data/x_train.npy', df['x_train'])
    # np.save('../data/y_train.npy', df['y_train'])
    # return

    x_train = np.load('../data/x_train.npy')
    x_train = x_train.tolist()
    y_train = np.load('../data/y_train.npy')
    y_train = y_train.tolist()

    # dev_sample_index = -1 * int(0.1 * float(len(y_train)))
    dev_sample_index = -1 * int(0.2 * float(len(y_train)))
    # 划分训练集和验证集
    # x_train, x_val = x_train[:dev_sample_index], x_train[dev_sample_index:]
    x_train, x_val = x_train, x_train[dev_sample_index:]
    # y_train, y_val = y_train[:dev_sample_index], y_train[dev_sample_index:]
    y_train, y_val = y_train, y_train[dev_sample_index:]
    # print('train:{}, val:{}, all:{}'.format(len(y_train), len(y_val), df.shape[0]))

    model.train(x_train, y_train, x_val, y_val)


def predict():
    test = pd.read_csv(config.test_path)
    # cols = ['patient_id', 'age', 'sex', 'adult_bmi', 'child_weight', 'child_height', 'audio_and_txt_files_path']
    # df = df.ix[:, cols]
    # df['x_test'] = df.apply(lambda row: deal_x(row), axis=1)
    # np.save('../data/x_test.npy', df['x_test'])
    # return
    x_test = np.load('../data/x_test.npy')
    x_test = np.asarray(x_test.tolist())
    preds = model.predict(x_test)
    test['diagnosis'] = [label_dict_res[var] for var in preds]
    test['patient_id'] = test['patient_id'] - 769
    test[['patient_id', 'diagnosis']].to_csv('../data/pred.csv', index=None, header=None)


def main(op):
    if op == 'train':
        train()
    elif op == 'predict':
        predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--operation", default='train', type=str, help="operation")
    parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
    parser.add_argument("-e", "--EPOCHS", default=8, type=int, help="train epochs")
    args = parser.parse_args()
    config.batch_size = args.BATCH
    config.epochs_num = args.EPOCHS
    from model.cnn_model import CNNModel

    model = CNNModel()
    main(args.operation)
