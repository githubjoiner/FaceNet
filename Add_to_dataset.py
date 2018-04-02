from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image as im
import tensorflow as tf
import numpy as np
import facenet
import pickle
import os

def imgs_to_embedding(imgs_dir):
    # imgs_dir：需要编码的多个人的图片路径
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model('facenetmodel')
            # Get input and output tensors
            # sess = tf.Session()
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embed = []
            for img_dir in imgs_dir:
                image = np.array(im.open(img_dir).resize((160, 160))).reshape((1, 160, 160, 3)).astype('float32')
                feed_dict = {images_placeholder: image, phase_train_placeholder: False}
                temp = sess.run(embeddings, feed_dict=feed_dict)
                embed.append(temp)
            return embed
def add_some_to_dataset(root_dir, dataset_dir='face_dataset.pkl'):
    #root_dir:所有需要存入数据库的图片的根目录

    list_dir = os.listdir(root_dir)
    file_list = []
    name_list = []
    for total_dir in list_dir:
        file_list.append(os.path.join(root_dir, total_dir))
        name_list.append(total_dir.split('.')[0])

    embeddings = imgs_to_embedding(file_list)

    with open(dataset_dir, 'rb') as rf:
        dic = pickle.load(rf)
    for name, embedding in zip(name_list, embeddings):
        dic[name] = embedding  # 将人脸编码添加进数据库
    with open(dataset_dir, 'wb') as wf:
        pickle.dump(dic, wf)
    print('%s成功添加进数据库！' % name_list)
if __name__ == '__main__':
    root_dir = ''   # 所有需要存入数据库的图片的根目录
    add_some_to_dataset(root_dir)