import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        GS = np.zeros((self.wndw_size, self.wndw_size))
        for i in range(self.wndw_size):
            for j in range(self.wndw_size):
                GS[i,j] = np.exp( np.divide(np.square(i-self.pad_w)+np.square(j-self.pad_w),-2*self.sigma_s*self.sigma_s) )

        padded_guidance = padded_guidance.astype('float64')
        padded_guidance /= 255

        padded_img = padded_img.astype('float64')
        output = np.zeros(img.shape)

        for i in range(self.pad_w, padded_guidance.shape[0]-self.pad_w):
            for j in range(self.pad_w, padded_guidance.shape[1]-self.pad_w):
                TP = padded_guidance[i,j]
                TQ = padded_guidance[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1]
                Power = np.divide(np.square(TP-TQ), -2*self.sigma_r*self.sigma_r)
                if len(Power.shape)==3:
                    Power = Power.sum(axis=2)  
                GR=np.exp(Power)
                G =np.multiply(GS, GR)
                GW =G.sum(axis=1).sum(axis=0)
                IQ=padded_img[i-self.pad_w:i+self.pad_w+1, j-self.pad_w:j+self.pad_w+1]        
                for k in range(img.shape[2]):
                    output[i-self.pad_w, j-self.pad_w, k] = np.multiply(G,IQ[:,:,k]).sum(axis=1).sum(axis=0)/GW
        output.astype(np.uint8)

        return np.clip(output, 0, 255).astype(np.uint8)