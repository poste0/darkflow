from darkflow.net.build import TFNet
import cv2
from time import time as timer
import json
options = {"pbLoad": "/home/sstepanenko/darkflow_TRT/darkflow/built_graph/tiny-yolo-voc.pb", "metaLoad": "/home/sergey/darkflow/built_graph/tiny-yolo-voc.meta", "gpu": 1.0}
file = open("/home/sstepanenko/darkflow_TRT/logs.txt" , 'w')
filee = open("/home/sstepanenko/darkflow_TRT/logss.txt" , 'w')
tfnet = TFNet(options)

result = []
s = timer()
reader = cv2.VideoCapture("/home/sstepanenko/darkflow_TRT/darkflow/recordd.avi")
k = 0
while reader.isOpened():
    _ , imgcv = reader.read()
    if imgcv is None:
        break
    r = tfnet.return_predict(imgcv)
    if len(r) != 0:
        print(r)
        json.dump(r , file)
        file.write('\n')
    if k % 30 == 0 :
        time = timer() - s
        print(30 / time)
        filee.write(str(30 / time))
        filee.write('\n')
        s = timer()
        k = 0
    k += 1



print(result)
