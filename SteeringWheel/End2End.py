import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.core.protobuf import saver_pb2
import Project.SteeringWheel.Preprocessing as ps 
import Model


class End2End(ps.ValueClass):
    """
    End2End : Create 11/19/2020 by ratchanonth
    lib for Pi3 
    """
    def __new__(self):
        self.sess = tf.InteractiveSession()
        self.train_vars = tf.trainable_variables()

    def __init__(self,**kawge):
        self.LOGDIR=kawge["LOGDIR"]
        self.EPOCHS=kawge["EPOCHS"]
        self.BATCH_SIZE=kawge["BATCH_SIZE"]
        self.driving=kawge["driving"]
        if kawge["L2NormConst"] is not None :
            self.L2NormConst= kawge["L2NormConst"]

    @property
    def _loss(self):
        return  tf.reduce_mean(tf.square(tf.subtract(Model.y_, Model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in self.train_vars]) * self.L2NormConst

    @property
    def _train_step(self):
        return   tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def train(self):
        self.sess.run(tf.global_variables_initializer())

        # create a summary to monitor cost tensor
        tf.summary.scalar("loss", self.loss)

        # merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V2)

        # op to write logs to Tensorboard
        logs_path = './logs'
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        
        for epoch in range(self.EPOCHS):
            for i in range(int(self.driving.GetNumberImage / self.BATCH_SIZE)):
                xs, ys =self.driving.LoadTrainBatch(self.BATCH_SIZE)
                self._train_step.run(feed_dict={Model.x: xs, Model.y_: ys, Model.keep_prob: 0.8})
                if i % 10 == 0:
                    xs, ys =self.driving.LoadValBatch(self.BATCH_SIZE)
                    loss_value=self._loss.eval(feed_dict={Model.x:xs, Model.y_: ys, Model.keep_prob: 1.0})
                    print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * self.BATCH_SIZE + i, loss_value))
epochs = 30
batch_size = 100
driving=ps.ReadData("test.txt",0.8,0.2)
e2e=End2End(driving=driving,LOGDIR='./save',EPOCHS=30,BATCH_SIZE=100,L2NormConst=0.001)
e2e.train()