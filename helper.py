import os
import numpy as np
import json
import cv2

class MHIProcessor:
    '''
    Process MHI as inputs of Fall Detector model
    '''
    def __init__(self, dim=128, threshold=0.1, interval=2, duration=40):
        # initialize MHI params
        self.index = 0
        self.dim = dim
        self.threshold = threshold
        self.interval = interval
        self.duration = duration
        self.decay = 1 / self.duration
        
        #initialize frames
        self.mhi_zeros = np.zeros((dim, dim))        
        
    
    def process(self, frame, save_batch=True):
        self.index += 1

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.index == 1:
            self.prev_frame = cv2.resize(frame,(self.dim, self.dim),
                                         interpolation=cv2.INTER_AREA)
            self.prev_mhi = self.mhi_zeros
            
        if self.index % self.interval == 0:
            frame = cv2.resize(frame,(self.dim, self.dim),
                                         interpolation=cv2.INTER_AREA)
            diff = cv2.absdiff(self.prev_frame, frame)
            binary = (diff >= (self.threshold * 255)).astype(np.uint8)
            mhi = binary + (binary == 0) * np.maximum(self.mhi_zeros,
                                                      (self.prev_mhi - self.decay))
            # update frames
            self.prev_frame = frame
            self.prev_mhi = mhi
            
            if self.index >= (self.duration * self.interval):
                img = cv2.normalize(mhi, None, 0.0, 255.0, cv2.NORM_MINMAX)
                if save_batch:
                    return img
                else:
                    flag, encode_img = cv2.imencode('.png', img)
                    if flag:
                        return bytearray(encode_img)
                
        return None