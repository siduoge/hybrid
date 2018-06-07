import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
import shutil
import random
import time
import pickle


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


class param():

    batchSize = 32
    inputDim = 22
    outputDim = 2
    hiddenDim = 128
    classes = 2
    length = 5
    row = 2
    rnn = 'lst'

class RNN():


    def __init__(self,config,testPath,trainPath,dicPath,word2VecPath):
        """
        init the RNN input param
        :param config: the parma of RNN network
        :param testPath: test dataset path
        :param trainPath:  train dataset path
        """
        self.config = config
        self.input = tf.placeholder(dtype=tf.float32, shape=(None, self.config.length, self.config.inputDim))
        self.output = tf.placeholder(dtype=tf.int32, shape=(None, self.config.outputDim))
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.testPath = testPath
        self.trainPath = trainPath
        self.standard = self.initStandrdScaler(self.trainPath)
        self.initWord2Vec(dicPath,word2VecPath)

    def initStandrdScaler(self,path):
        """
        standard the data
        :param path: train dataset path
        :return:
        """
        self.name = ["query_date", "subappid", "label", 'teg_evil_result_level', 'found', 'id_found', 'cdg_qq_black',
                'cdg_wx_black','cdg_other_black', 'cdg_id_auth', 'cdg_gf_black', 'riskcode_1_level', 'riskcode_2_level',
                'riskcode_3_level','riskcode_4_level', 'riskcode_5_level', 'riskcode_6_level', 'riskcode_7_level', 'riskcode_8_level', \
                'riskcode_301_level', 'riskcode_503_level']
        data = pd.read_csv(path)
        data = data.drop(columns=self.name)
        data = data.fillna(0)
        scaler = StandardScaler().fit(data)
        return scaler


    def initWord2Vec(self,dicPath,word2vecPath):
        self.dictApp = {}
        with open(dicPath, "r") as fr:
            for line in fr.readlines():
                line = line.strip()
                if len(line) != 0:
                    self.dictApp[line] = len(self.dictApp)
        #reverdictApp = dict(zip(self.dictApp.values(), self.dictApp.keys()))
        with open(word2vecPath, "rb") as f:
            self.w = pickle.load(f)

    def tebsorboardInit(self,sess):
        """
        init the tensorboard , testSet, TrainSet, ValidationSet
        :param sess: session , record the graph
        :return: None
        """
        self.logdir = "./tensorboard/RNN/"+ time.ctime(time.time())
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

    def lstm_cell(self,dim):
        """
        lstm cell
        :return: lstm
        """
        return tf.contrib.rnn.BasicLSTMCell(dim, state_is_tuple=True)

    def gru_cell(self,dim):
        """
        gru cell
        :return: gru
        """
        return tf.contrib.rnn.GRUCell(dim)

    def dropout(self,dim):
        """
         add a dropout function for each layer
        :return:
        """
        if (self.config.rnn == 'lstm'):
            cell = self.lstm_cell(dim)
        else:
            cell = self.gru_cell(dim)
        return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

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
        with tf.variable_scope("rnn"):
            cells = [self.dropout(self.config.hiddenDim) for _ in range(self.config.row)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=self.input, dtype=tf.float32)
        with tf.variable_scope("averageLayer"):
            averageLayer = tf.reduce_mean(outputs, axis=1)

        layer3 = self.nn_layer(averageLayer, self.config.hiddenDim, self.config.outputDim, "layer3")
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

    def validationDataset(self,path):
        """
        create the validation dataset for  RNN
        :param path: the data path
        :return: (data,label)
        """
        dataSet = pd.read_csv(path)
        dataSet = dataSet.fillna(0)
        labelSet = dataSet.label.astype(np.int32).as_matrix()
        appid = dataSet.subappid.astype(np.int32).astype(str).as_matrix()
        appidVec = [self.w[self.dictApp[i]] for i in appid]
        appidVec = np.array(appidVec)
        data = dataSet.drop(columns=self.name)
        data = self.standard.transform(data.as_matrix())
        data = np.c_[data, appidVec]
        data = np.reshape(data, newshape=(-1, self.config.length, self.config.inputDim))
        label = []
        for index, x in enumerate(labelSet):
            if (index + 1) % self.config.length == 0:
                label.append(x)
        return data, label

    def creatBatchTrain(self,data,label):
        """
        create  batch train  for  DNN
        :return: (batch of data, batch of label)
        """
        data = np.array(data)
        label = np.array(label)
        length = data.shape[0]
        num = length // self.config.batchSize
        arr = [i for i in range(1, num)]
        random.shuffle(arr)
        for i in arr:
            yield data[(i-1) * self.config.batchSize: i * self.config.batchSize],\
                  label[(i-1) * self.config.batchSize:i * self.config.batchSize]



    def train(self,epoch=5):
        self.optimize(self.network())

        with tf.Session() as sess:
            self.tebsorboardInit(sess)
            tf.global_variables_initializer().run()
            testData, testLabel1 = self.validationDataset(self.testPath)
            trainData, trainLabel = self.validationDataset(self.trainPath)
            testLabel = tf.one_hot(testLabel1, self.config.outputDim).eval(session=sess)
            trainLabel = tf.one_hot(trainLabel, self.config.outputDim).eval(session=sess)

            num = 0

            start = time.time()
            for i in range(epoch):
                for trainX, trainY in self.creatBatchTrain(trainData,trainLabel):
                    tf.local_variables_initializer().run()
                    summary,_, stepLoss = sess.run([self.merged,self.train_step, self.cross_entropy], feed_dict={self.input: trainX, self.output: trainY,self.keep_prob:1})
                    self.train_writer.add_summary(summary,num)
                    if num % 50 == 0:
                        print("step:", num, " ,loss:", stepLoss)
                        tf.local_variables_initializer().run()
                        predict,summary,acc,auc = sess.run([self.predict,self.merged,self.accuracy,self.auc], feed_dict={self.input: testData, self.output: testLabel,self.keep_prob:1})
                        print("test accuracy:%f; test auc: %f"  %(acc,auc[1]))
                        ks = metrics_ks(np.array(testLabel1), predict[:, 1])
                        print("ks value: %f" % ks)
                        self.writeParam(ks)
                        self.test_writer.add_summary(summary, num)
                        tf.local_variables_initializer().run()
                        summary= sess.run(self.merged,feed_dict={self.input: trainData, self.output: trainLabel, self.keep_prob: 1})
                        self.val_writer.add_summary(summary, num)
                    num += 1
            print("time: %f" %(time.time()-start))
            self.train_writer.close()
            self.test_writer.close()
            self.val_writer.close()

if __name__=="__main__":
    config = param()
    test = RNN(config,"../water_info/data/3test_ba.csv","../water_info/data/3train_ba.csv","../water_info/data/subappid.csv","../water_info/data/w2.pickle")
    test.train(epoch=1)


