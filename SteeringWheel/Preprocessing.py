import cv2
import random 
import numpy as np 
import math 

class ValueClass(object):
    """
    class value
    """
    #pointer to the end of last batch
    TRAIN_BATCH_POINTER=0
    VAL_BATCH_POINTER=0
    
    XS=[]
    YS=[]

    X_OUT = []
    Y_OUT = []

    LOGDIR = None
    L2NormConst=0.001
    EPOCHS=30
    BATCH_SIZE=100
    
    DATA=dict()

    def __init__(self):
        pass

    @property
    def GetNumberImage(self):
        #count of image 
        return self.XS.__len__()
    
    
    def ZipData(self):
        # shuffle list of image 
        return list(zip(self.XS,self.YS))


class ReadData(ValueClass):
    """
    docstring
    """
    

    def __init__(self,FileName,PercenTrain,PercenValidate):
        self.PercenTrain=PercenTrain
        self.PercenValidate=PercenValidate
        self.__OpenFile(FileName)
    
    def __OpenFile(self,FileName):
        """
        FileName : Path location of File 
        """
        with open(FileName) as data:
            for line in data:
               self.XS.append("driving_dataset/"+line.split()[0])
               self.YS.append(math.degrees(int(line.split()[1])))
    
    @property
    def TrainSplit(self):
        random.shuffle(self.ZipData())
        xs,ys=zip(*self.ZipData())
        train_xs=xs[:int(len(xs)*self.PercenTrain)]
        train_ys=ys[:int(len(xs)*self.PercenTrain)]

        val_xs=xs[-int(len(xs)*self.PercenValidate):]
        val_ys=ys[-int(len(xs)*self.PercenValidate):]

        num_train=len(train_xs)
        num_val=len(val_xs)

        self.DATA= {
            "train_xs":train_xs,
            "train_ys":train_ys,
            "val_xs":val_xs,
            "val_xs":val_xs,
            "num_train":num_train,
            "num_val":num_val
        }
    
    def LoadTrainBatch(self,batch_size):
        for i in range(batch_size):
            self.X_OUT.append(cv2.resize(cv2.imread(self.DATA["train_xs"][(self.TRAIN_BATCH_POINTER + i) % self.DATA["num_train"]])[-150:], (200, 66)) / 255.0)
            self.Y_OUT.append(self.DATA["train_ys"][self.DATA["train_xs"][(self.TRAIN_BATCH_POINTER + i) % self.DATA["num_train"]]])
        self.TRAIN_BATCH_POINTER+=batch_size

        return self.X_OUT,self.Y_OUT
    
    def LoadValBatch(self,batch_size):
        for i in range(batch_size):
            self.X_OUT.append(cv2.resize(cv2.imread(self.DATA["val_xs"][(self.VAL_BATCH_POINTER + i) % self.DATA["num_train"]])[-150:], (200, 66)) / 255.0)
            self.Y_OUT.append(self.DATA["val_xs"][self.DATA["val_xs"][(self.VAL_BATCH_POINTER + i) % self.DATA["num_train"]]])
        self.VAL_BATCH_POINTER+=batch_size
        
        return self.X_OUT,self.Y_OUT