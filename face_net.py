from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
from PIL import Image as im
import tensorflow as tf
import numpy as np
import facenet
import pickle
import os

def haveFaces(image_name):
    # 返回是否检测到人脸
    img = cv2.imread(image_name)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img# if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)# 1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
    result = []
    for (x, y, width, height) in faces:
        result.append((x, y, x+width, y+height))

    if result:
        #将人脸保存在save_dir目录下。
        #Image模块：Image.open获取图像句柄，crop剪切图像(剪切的区域就是detectFaces返回的坐标)，save保存。
        save_dir = './picture/temp_faces'
        count = 0
        for (x1, y1, x2, y2) in result:
            file_name = os.path.join(save_dir, str(count)+".jpg")
            im.open(image_name).crop((x1, y1, x2, y2)).save(file_name)
            count += 1
        return True
    else:
        return False

def img_to_enbedding(img_dir):
    # img_dir:图片的路径
    # 输出图片的编码：1*128
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model('facenetmodel')
            # Get input and output tensors
            # sess = tf.Session()
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            image = np.array(im.open(img_dir).resize((160, 160))).reshape((1, 160, 160, 3)).astype('float32')
            feed_dict = {images_placeholder: image, phase_train_placeholder: False}
            embed = sess.run(embeddings, feed_dict=feed_dict)
            return embed

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

def cal_simalar(embed1, embed2):
    # embed1、embed2 : 需要计算相似度的两个向量，相似度越高，输出值越小
    return np.sum(np.power(embed2-embed1, 2))

def who_is(embedding, dataset_dir='face_dataset.pkl', threshold = 0.1):
    # embedding: 需要比对的人的编码
    # threshold：阈值，若相似性低于阈值，则说明比对成功。
    with open(dataset_dir, 'rb') as rf:
        dic = pickle.load(rf)
    min_dis = 999
    for key in dic:
        distance = cal_simalar(dic[key], embedding)
        if distance < min_dis:
            min_dis = distance
            name = key

    if min_dis < threshold:
        return name
    else:
        return None

def main():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model('facenetmodel')
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            # add_one_to_dataset('古俊波', embed)
            print('连续按“q”退出！')
            vc = cv2.VideoCapture(0)  # 读入视频文件
            c = 1
            if vc.isOpened():  # 判断是否正常打开
                rval, frame = vc.read()
            else:
                rval = False
            timeF = 30  # 视频帧计数间隔频率
            cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            while rval:  # 循环读取视频帧
                rval, frame = vc.read()
                # cv2.imshow('face_recognition', frame)
                #人脸检测
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rect = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=9, minSize=(50, 50),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
                for x, y, z, w in rect:
                    cv2.rectangle(frame, (x, y), (x + z, y + w), (0, 0, 255), 2)
                cv2.imshow('frame', frame)
                # 判断是否按下q键
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # 清除缓存图片
                    if os.path.exists('./picture/temp.jpg'):
                        os.remove('./picture/temp.jpg')

                    if os.listdir('./picture/temp_faces'):
                        for i in os.listdir('./picture/temp_faces'):
                            os.remove(os.path.join('./picture/temp_faces', i))
                    break
                # 每隔timeF帧进行存储操作
                if (c % timeF == 0):
                    cv2.imwrite('./picture/temp.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # 存储为图像
                    people = []
                    exit_face = haveFaces('./picture/temp.jpg')
                    if exit_face:
                        for pic in os.listdir('./picture/temp_faces'):
                            image = np.array(im.open('./picture/temp_faces/%s' % pic).resize((160, 160))).reshape((1, 160, 160, 3)).astype('float32')
                            feed_dict = {images_placeholder: image, phase_train_placeholder: False}
                            embed = sess.run(embeddings, feed_dict=feed_dict)
                            the_one = who_is(embed)
                            people.append(the_one)
                            os.remove('./picture/temp_faces/%s' % pic)
                    else:
                        image = np.array(im.open('./picture/temp.jpg').resize((160, 160))).reshape(
                            (1, 160, 160, 3)).astype('float32')
                        feed_dict = {images_placeholder: image, phase_train_placeholder: False}
                        embed = sess.run(embeddings, feed_dict=feed_dict)
                        the_one = who_is(embed)
                        people.append(the_one)
                    print(people)

                c = c + 1
                cv2.waitKey(1)
            vc.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

