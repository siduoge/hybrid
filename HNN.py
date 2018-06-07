import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import sys
sys.path.append('/home/siduoge/PycharmProjects/tf_mix/code')
from DeepNN import DNN, param as paDNN
from RecurrentNN import RNN, param as paRNN
import time
import shutil
import os
import random


def metrics_ks(label, predict):
    good = 1 - label
    bad = label
    d = {'good': good, 'bad': bad, 'predict': np.array(predict)}
    data = pd.DataFrame(d)
    data['bucket'] = pd.qcut(data.predict, 10, duplicates='drop')
    grouped = data.groupby('bucket', as_index=False)
    stats = pd.DataFrame(grouped.sum().bad, columns=['bad'])
    stats['good'] = grouped.sum().good
    stats['total'] = stats.bad + stats.good
    stats['badRate'] = (stats.bad / data.bad.sum()).cumsum()
    stats['goodRate'] = (stats.good / data.good.sum()).cumsum()
    stats['ksVal'] = np.abs((stats.bad / data.bad.sum()).cumsum() - (stats.good / data.good.sum()).cumsum())
    return max(stats.ksVal.values)

class paraHNN():
    """
    param of HNN
    """
    inputDim=128
    batchSize=32
    hiddenDim1=128
    hiddenDim2=128
    hiddenDim3=128
    outputDim=2



class HNN():
    def __init__(self,paraHNN,paraDNN,paraRNN,dnn,rnn,testDnn,trainDnn,testRnn,trainRnn,dictPath,weightPath):
        """
        init the hybrid neutral network
        :param dnn:  deep neutral network
        :param rnn:  recurrent neutral network
        :return: None
        """
        self.para = paraHNN
        self.dnn = dnn(paraDNN,testDnn,trainDnn)
        self.rnn = rnn(paraRNN,testRnn,trainRnn,dictPath,weightPath)
        self.DNNinput = self.dnn.input
        self.RNNinput = self.rnn.input
        self.dnnProb = self.dnn.keep_prob
        self.rnnProb = self.rnn.keep_prob
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.output = tf.placeholder(tf.float32,shape=(None,self.para.outputDim))

    def tebsorboardInit(self,sess):
        """
        init the tensorboard , testSet, TrainSet, ValidationSet
        :param sess: session , record the graph
        :return: None
        """
        self.logdir = "./tensorboard/HNN/"+ time.ctime(time.time())
        if os.path.exists(self.logdir):
            shutil.rmtree(self.logdir)
        os.makedirs(self.logdir)
        self.merged = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(self.logdir+'/train',sess.graph)
        self.val_writer = tf.summary.FileWriter(self.logdir + '/val')
        self.test_writer = tf.summary.FileWriter(self.logdir+ '/test')

    def variable_summaries(self, var):
        """

        :param var:  the variable which is need to be recorded
        :return: None
        """
        with tf.variable_scope(None,'summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.variable_scope(None,'stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


    def weight_variable(self,shape):
        """
        Create a weight variable with appropriate initialization.
        :param shape: the shape of weight
        :return: tf.variable
        """
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        """
        Create a bias variable with appropriate initialization.
        :param shape: the shape of bias
        :return: tf.variable
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def nn_layer(self,input, input_dim, output_dim, name, act=tf.nn.leaky_relu):
        """
        Reusable code for making a simple neural net layer the default activation function is leaky_relu
        :param input_tensor: input tensor
        :param input_dim: the dimension of input tenosr
        :param output_dim: the dimension of output tensor
        :param layer_name: the name of layer
        :param act: activation function
        :return: nonlinear tensor
        """
        with tf.variable_scope(None,name):
            with tf.variable_scope(None,'weights'):
                w = self.weight_variable([input_dim, output_dim])
                self.variable_summaries(w)
            with tf.variable_scope(None,"bias"):
                b = self.bias_variable([output_dim])
                self.variable_summaries(b)
            with tf.variable_scope(None,"act"):
                pre_layer1 = tf.matmul(input, w) + b
                tf.summary.histogram('act', pre_layer1)
                layer1 = act(pre_layer1)
            with tf.variable_scope(None,"drop_out1"):
                dropOutLayer1 = tf.contrib.layers.dropout(layer1, self.keep_prob)
        return  dropOutLayer1

    def network(self):
        layer0 = tf.concat([self.dnn.network(), self.rnn.network()], axis=1)
        layer1 = self.nn_layer(layer0, self.para.inputDim, self.para.hiddenDim1, "layer1")
        layer2 = self.nn_layer(layer1, self.para.hiddenDim1, self.para.hiddenDim2, "layer2")
        layer3 = self.nn_layer(layer2, self.para.hiddenDim2, self.para.outputDim, "layer3")
        return layer3

    def optimize(self,input):
        """
         optimize the loss and calculate the accuracy, auc and ks
         :param input: the last layer output
         :return: None
         """
        with tf.name_scope("cross_entropy"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.output, logits=input))
            tf.summary.scalar('cross_entropy', self.cross_entropy)

        with tf.name_scope("optimaze"):
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(1e-3, global_step, decay_steps=5000, decay_rate=0.90,
                                                       staircase=True)
            self.train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cross_entropy)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.output, 1), tf.argmax(tf.nn.softmax(input), 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        with tf.variable_scope(None,"auc"):
            self.auc = tf.metrics.auc(self.output,tf.nn.softmax(input))
            tf.summary.scalar('auc', self.auc[1])

        with tf.name_scope("predict"):
            self.predict = tf.nn.softmax(input)

    def writeParam(self, pro):
        name = os.path.join("%s/param.txt" % self.logdir)
        with open(name, "a") as fw:
            fw.write(str(pro) + "\n")

    def validationDataset(self,path1,path2):
        """
        get the validation data
        :param path1: path of dnn
        :param path2: path of rnn
        :return: (dnnData,rnnData,label)
        """
        DnnData,DnnLabel = self.dnn.validationDataset(path1)
        RnnData,RnnLabel = self.rnn.validationDataset(path2)
        return DnnData, RnnData,RnnLabel

    def creatBatchTrain(self,dnnData,rnnData,label):
        """

        :param dnnData:
        :param rnnData:
        :param label:
        :return:
        """
        dnnData = np.array(dnnData)
        rnnData = np.array(rnnData)
        label = np.array(label)
        length = dnnData.shape[0]
        num = length // self.para.batchSize
        arr = [i for i in range(1, num)]
        random.shuffle(arr)
        for i in arr:
            yield dnnData[(i - 1) * self.para.batchSize: i * self.para.batchSize], \
                  rnnData[(i - 1) * self.para.batchSize: i * self.para.batchSize],\
                  label[(i - 1) * self.para.batchSize:i * self.para.batchSize]

    def train(self,epoch=5):
        self.optimize(self.network())

        with tf.Session() as sess:
            self.tebsorboardInit(sess)
            tf.global_variables_initializer().run()
            testDnnData, testRnnData,testLabel1 = self.validationDataset(self.dnn.testPath,self.rnn.testPath)
            trainDnnData, trainRnnData,trainLabel = self.validationDataset(self.dnn.trainPath,self.rnn.trainPath)
            testLabel = tf.one_hot(testLabel1, self.para.outputDim).eval(session=sess)
            trainLabel = tf.one_hot(trainLabel, self.para.outputDim).eval(session=sess)

            num = 0

            start = time.time()
            for i in range(epoch):
                for trainDnnX, trainRnnX,trainY in self.creatBatchTrain(trainDnnData,trainRnnData,trainLabel):
                    tf.local_variables_initializer().run()
                    summary,_, stepLoss = sess.run([self.merged,self.train_step, self.cross_entropy], feed_dict={self.DNNinput: trainDnnX, self.RNNinput:trainRnnX,self.output: trainY,self.keep_prob:1,self.dnnProb:1,self.rnnProb:1})
                    self.train_writer.add_summary(summary,num)
                    if num % 50 == 0:
                        print("step:", num, " ,loss:", stepLoss)
                        tf.local_variables_initializer().run()
                        predict,summary,acc,auc = sess.run([self.predict,self.merged,self.accuracy,self.auc], feed_dict={self.DNNinput: testDnnData, self.RNNinput:testRnnData,self.output: testLabel,self.keep_prob:1,self.dnnProb:1,self.rnnProb:1})
                        print("test accuracy:%f; test auc: %f"  %(acc,auc[1]))
                        ks = metrics_ks(np.array(testLabel1), predict[:, 1])
                        print("ks value: %f" % ks)
                        self.writeParam(ks)
                        self.test_writer.add_summary(summary, num)
                        tf.local_variables_initializer().run()
                        summary= sess.run(self.merged,feed_dict={self.DNNinput: trainDnnData, self.RNNinput:trainRnnData,self.output: trainLabel, self.keep_prob: 1,self.dnnProb:1,self.rnnProb:1})
                        self.val_writer.add_summary(summary, num)
                    num += 1
            print("time: %f" %(time.time()-start))
            self.train_writer.close()
            self.test_writer.close()
            self.val_writer.close()

if __name__=="__main__":
    paramDNN = paDNN()
    paramDNN.outputDim=64
    paramRNN = paRNN()
    paramRNN.outputDim=64
    paramHNN = paraHNN()
    paramHNN.inputDim=paramDNN.outputDim+paramRNN.outputDim
    paramHNN.batchSize =paramRNN.batchSize = paramDNN.batchSize = 32
    testDnnPath = "../water_info/data/3test_full.csv"
    trainDnnPath = "../water_info/data/3train_full.csv"
    testRnnPath = "../water_info/data/3test_ba.csv"
    trainRnnPath = "../water_info/data/3train_ba.csv"
    dictPath = "../water_info/data/subappid.csv"
    weightPath="../water_info/data/w2.pickle"
    test = HNN(paramHNN,paramDNN,paramRNN,DNN,RNN,testDnnPath,trainDnnPath,testRnnPath,trainRnnPath,dictPath,weightPath)
    test.train(epoch=10)
