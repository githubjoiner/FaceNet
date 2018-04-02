# FaceNet
The model provided by David Sandberg ,thanks ! Here is his homepage:https://github.com/davidsandberg

Using the triplet loss as loss function,you can encoding your pictures as vectors through the Network.
# step1
Download this model through : https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk and save them in 'models'
# step2
Before projecting, you should add the picture which you want to recognize to the dataset through Add_to_dataset.py .
# step3
Run face_net.py to start your project ,it needs a few minutes to load the model，please be patient .
# pay attention
If you want to improve the accuracy，you can reduce the threshold which in function 'who_is' in face_net.py .Done this,your model's accuracy will be increased,but sometimes it couldn't recognize you or others well.
