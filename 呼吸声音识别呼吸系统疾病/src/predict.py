# coding=utf-8
# author=yphacker

import pandas as pd
from flyai.dataset import Dataset

from model import Model


def eval():
    train = pd.read_csv('data/input/dev.csv')
    cols = ['patient_id', 'age', 'sex', 'adult_bmi', 'child_weight', 'child_height', 'audio_and_txt_files_path']
    x_test = train.ix[:, cols]

    for i, row in x_test.iterrows():
        p = model.predict(**row)
        print(list(train[train['patient_id'] == row['patient_id']]['diagnosis']), p)
        break


def predict():
    x_test = pd.read_csv('data/test.csv')
    cols = ['patient_id', 'age', 'sex', 'adult_bmi', 'child_weight', 'child_height', 'audio_and_txt_files_path']
    x_test = x_test.ix[:, cols]
    rows = []
    for i, row in x_test.iterrows():
        rows.append(row)
        # p = model.predict(**row)
        # print(data.to_categorys(p))
        # ans.append(data.to_categorys(p))
    ans = model.predict_all(rows)

    x_test['diagnosis'] = ans
    x_test['patient_id'] = x_test['patient_id'] - 769
    x_test[['patient_id', 'diagnosis']].to_csv('pred.csv', index=None, header=None)


if __name__ == '__main__':
    data = Dataset()
    model = Model(data, do_train=False)
    eval()
    # predict()
