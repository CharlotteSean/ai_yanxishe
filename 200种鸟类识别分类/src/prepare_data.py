# coding=utf-8
"""Prepare data.
"""
import os

# lines = [line.strip().split() for line in open('classes.txt')]

# # make dirs
# name = dict()
# for line in lines:
#     output_dir = 'val_set/{}'.format(line[1])
#     name[line[0]] = line[1]
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

# lines = [line.strip().split(',') for line in open('val_pname_to_index.csv')]

# # move files
# for line in lines:
#     file_path = 'val_set/{}'.format(line[0])
#     if not os.path.exists(file_path):
#         print(file_path)
#         continue
#     os.rename(file_path, 'val_set/{}/{}'.format(name[line[1]], line[0]))

# test
for i in range(1, 1769):
    output_dir = 'test_set/{:04}'.format(i)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = 'test_set/{}.jpg'.format(i)
    if not os.path.exists(file_path):
        print(file_path)
        continue
    os.rename(file_path, 'test_set/{:04}/{}.jpg'.format(i, i))

