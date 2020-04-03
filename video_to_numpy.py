import cv2
import collections
import numpy as np

def video_to_numpy(video_path,save_path,temporal_dimension=10):
    """
    Convert video to a numpy array w/ temporal dimension
    :param video_path: input video path
    :param save_path: output path
    :param temporal_dimension: temporal dimension for time distributed layer
    :return:
    """
    fname=video_path
    if video_path.rfind('/')!=-1:
        fname=video_path[video_path.rfind('/')+1:]
    frame_buffer=collections.deque(maxlen=temporal_dimension)
    video=cv2.VideoCapture(video_path)
    out_arr=None
    while True:
        isok,frame=video.read()
        if not isok:
            break
        frame = cv2.resize(frame, (224,224), interpolation = cv2.INTER_AREA)
        if len(frame_buffer)==0:
            for i in range(temporal_dimension):
                frame_buffer.append(frame)
        frame_buffer.append(frame)
        if out_arr is None:
            out_arr=np.expand_dims(np.stack(frame_buffer,axis=0), axis=0)
        else:
            current_arr=np.expand_dims(np.stack(frame_buffer,axis=0), axis=0)
            #print('#',out_arr.shape,current_arr.shape)
            out_arr=np.concatenate([out_arr,current_arr],axis=0)
        # cv2.imshow('a',frame)
        # cv2.waitKey(100)
    np.save(save_path+fname,out_arr)
