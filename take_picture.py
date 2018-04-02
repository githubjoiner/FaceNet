import cv2
root_dir = 'picture/'
print('按“p”进行拍照,连按”q“退出！')
vc = cv2.VideoCapture(0) #读入视频文件
if vc.isOpened(): #判断是否正常打开
    rval , frame = vc.read()
else:
    rval = False
timeF = 5  #视频帧计数间隔频率
while rval:   #循环读取视频帧
    rval, frame = vc.read()
    cv2.imshow('test', frame)
    if cv2.waitKey(1) & 0xFF == ord('p'):
        cv2.imwrite(root_dir+"%s.jpg" % (input("请输入保存的图片的名称！")), frame)
        print('图片保存成功！')

        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vc.release()
cv2.destroyAllWindows()