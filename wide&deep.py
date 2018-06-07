import tensorflow as tf
import pandas as pd
import numpy as np
import random
import time
import os
import shutil
from DeepNN import DNN, param as paDNN
from RecurrentNN import RNN, param as paRNN
from HNN import HNN,paraHNN


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
    inputDim=252
    concatDim=380
    outputDim=2
    batchSize=32


class wideDeep():
    def __init__(self,config,hnn,paraHNN,paraDNN,paraRNN,dnn,rnn,testDnn,trainDnn,testRnn,trainRnn,dictPath,weightPath):
        self.para = config
        self.hnn = hnn(paraHNN,paraDNN,paraRNN,dnn,rnn,testDnn,trainDnn,testRnn,trainRnn,dictPath,weightPath)
        self.wideInput = tf.placeholder(tf.float32, shape=(None, self.para.inputDim))
        self.output = tf.placeholder(tf.float32,shape=(None,self.para.outputDim))
        self.name = ['teg_evil_result_level', 'found', 'id_found', 'cdg_qq_black',
                'cdg_wx_black','cdg_other_black', 'cdg_id_auth', 'cdg_gf_black', 'riskcode_1_level', 'riskcode_2_level',
                'riskcode_3_level','riskcode_4_level', 'riskcode_5_level', 'riskcode_6_level', 'riskcode_7_level', 'riskcode_8_level', \
                'riskcode_301_level', 'riskcode_503_level', 'age', 'sex', 'label']

    def tebsorboardInit(self, sess):
        """
        init the tensorboard , testSet, TrainSet, ValidationSet
        :param sess: session , record the graph
        :return: None
        """
        self.logdir = "./tensorboard/wideDeep/" + time.ctime(time.time())
        if os.path.exists(self.logdir):
            shutil.rmtree(self.logdir)
        os.makedirs(self.logdir)
        self.merged = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(self.logdir + '/train', sess.graph)
        self.val_writer = tf.summary.FileWriter(self.logdir + '/val')
        self.test_writer = tf.summary.FileWriter(self.logdir + '/test')

    def network(self):
        with tf.name_scope("concat"):
            input = tf.concat([self.hnn.network(),self.wideInput],axis=1)
        with tf.name_scope("weights"):
            w = tf.get_variable("wideW",shape=(self.para.concatDim,self.para.outputDim))
            b = tf.get_variable("biasb",shape=(self.para.outputDim))
            layer = tf.nn.leaky_relu(tf.matmul(input,w)+b)
        return layer
    def optimize(self,input):
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.output,logits=input))

        with tf.name_scope("opimaze"):
            global_step = tf.Variable(0)
            learning_rate = tf.train.exponential_decay(1e-3, global_step, decay_steps=50000, decay_rate=0.90,
                                                       staircase=True)
            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        with tf.name_scope("auc"):
            self.auc = tf.metrics.auc(self.output,tf.nn.softmax(input))
            tf.summary.scalar('auc', self.auc[1])

        with tf.name_scope("predict"):
            self.predict = tf.nn.softmax(input)

    def getCrossFeature(self,set1, set2, data1, data2):
        dic = self.crossFeatureDict(set1, set2)
        ret = np.zeros((data1.shape[0], len(dic)))
        for index, i in enumerate(data1):
            ret[index][self.getBoolFeature(dic, i, data2[index])] = 1
        return ret

    def crossFeatureDict(self,set1, set2):
        ret = {}
        for i in set1:
            for j in set2:
                ret[str(i) + "+" + str(j)] = len(ret)
        return ret

    def getBoolFeature(self,dic, num1, num2):
        name = str(num1) + "+" + str(num2)
        return dic[name]

    def getCrossFeatures(self,data):
        foundAndIdfound = set((-1, 1))
        sex = set((0, 1))
        riskCode = set((0, 1, 2, 3))
        riskCode5 = set((0, 2))
        riskCode7 = set((0, 3))
        feaOfRisk301AndRisk503 = self.getCrossFeature(riskCode, riskCode, data.riskcode_301_level.astype(np.int32),
                                                      data.riskcode_503_level.astype(np.int32))

        feaOfFoundAndrisk1 = self.getCrossFeature(foundAndIdfound, riskCode, data.found.astype(np.int32),
                                                  data.riskcode_1_level.astype(np.int32))
        feaOfFoundAndrisk2 = self.getCrossFeature(foundAndIdfound, riskCode, data.found.astype(np.int32),
                                                  data.riskcode_2_level.astype(np.int32))
        feaOfFoundAndrisk3 = self.getCrossFeature(foundAndIdfound, riskCode, data.found.astype(np.int32),
                                                  data.riskcode_3_level.astype(np.int32))
        feaOfFoundAndrisk4 = self.getCrossFeature(foundAndIdfound, riskCode, data.found.astype(np.int32),
                                                  data.riskcode_4_level.astype(np.int32))
        feaOfFoundAndrisk5 = self.getCrossFeature(foundAndIdfound, riskCode5, data.found.astype(np.int32),
                                                  data.riskcode_5_level.astype(np.int32))
        feaOfFoundAndrisk6 = self.getCrossFeature(foundAndIdfound, riskCode, data.found.astype(np.int32),
                                                  data.riskcode_6_level.astype(np.int32))
        feaOfFoundAndrisk7 = self.getCrossFeature(foundAndIdfound, riskCode7, data.found.astype(np.int32),
                                                  data.riskcode_7_level.astype(np.int32))
        feaOfFoundAndrisk8 = self.getCrossFeature(foundAndIdfound, riskCode, data.found.astype(np.int32),
                                                  data.riskcode_8_level.astype(np.int32))
        feaOfFoundAndrisk301 = self.getCrossFeature(foundAndIdfound, riskCode, data.found.astype(np.int32),
                                                    data.riskcode_301_level.astype(np.int32))
        feaOfFoundAndrisk503 = self.getCrossFeature(foundAndIdfound, riskCode, data.found.astype(np.int32),
                                                    data.riskcode_503_level.astype(np.int32))

        feaOfsexAndrisk1 = self.getCrossFeature(sex, riskCode, data.sex.astype(np.int32),
                                                data.riskcode_1_level.astype(np.int32))
        feaOfsexAndrisk2 = self.getCrossFeature(sex, riskCode, data.sex.astype(np.int32),
                                                data.riskcode_2_level.astype(np.int32))
        feaOfsexAndrisk3 = self.getCrossFeature(sex, riskCode, data.sex.astype(np.int32),
                                                data.riskcode_3_level.astype(np.int32))
        feaOfsexAndrisk4 = self.getCrossFeature(sex, riskCode, data.sex.astype(np.int32),
                                                data.riskcode_4_level.astype(np.int32))
        feaOfsexAndrisk5 = self.getCrossFeature(sex, riskCode5, data.sex.astype(np.int32),
                                                data.riskcode_5_level.astype(np.int32))
        feaOfsexAndrisk6 = self.getCrossFeature(sex, riskCode, data.sex.astype(np.int32),
                                                data.riskcode_6_level.astype(np.int32))
        feaOfsexAndrisk7 = self.getCrossFeature(sex, riskCode7, data.sex.astype(np.int32),
                                                data.riskcode_7_level.astype(np.int32))
        feaOfsexAndrisk8 = self.getCrossFeature(sex, riskCode, data.sex.astype(np.int32),
                                                data.riskcode_8_level.astype(np.int32))
        feaOfsexAndrisk301 = self.getCrossFeature(sex, riskCode, data.sex.astype(np.int32),
                                                  data.riskcode_301_level.astype(np.int32))
        feaOfsexAndrisk503 = self.getCrossFeature(sex, riskCode, data.sex.astype(np.int32),
                                                  data.riskcode_503_level.astype(np.int32))

        feaOfid_foundAndrisk1 = self.getCrossFeature(foundAndIdfound, riskCode, data.id_found.astype(np.int32),
                                                     data.riskcode_1_level.astype(np.int32))
        feaOfid_foundAndrisk2 = self.getCrossFeature(foundAndIdfound, riskCode, data.id_found.astype(np.int32),
                                                     data.riskcode_2_level.astype(np.int32))
        feaOfid_foundAndrisk3 = self.getCrossFeature(foundAndIdfound, riskCode, data.id_found.astype(np.int32),
                                                     data.riskcode_3_level.astype(np.int32))
        feaOfid_foundAndrisk4 = self.getCrossFeature(foundAndIdfound, riskCode, data.id_found.astype(np.int32),
                                                     data.riskcode_4_level.astype(np.int32))
        feaOfid_foundAndrisk5 = self.getCrossFeature(foundAndIdfound, riskCode5, data.id_found.astype(np.int32),
                                                     data.riskcode_5_level.astype(np.int32))
        feaOfid_foundAndrisk6 = self.getCrossFeature(foundAndIdfound, riskCode, data.id_found.astype(np.int32),
                                                     data.riskcode_6_level.astype(np.int32))
        feaOfid_foundAndrisk7 = self.getCrossFeature(foundAndIdfound, riskCode7, data.id_found.astype(np.int32),
                                                     data.riskcode_7_level.astype(np.int32))
        feaOfid_foundAndrisk8 = self.getCrossFeature(foundAndIdfound, riskCode, data.id_found.astype(np.int32),
                                                     data.riskcode_8_level.astype(np.int32))
        feaOfid_foundAndrisk301 = self.getCrossFeature(foundAndIdfound, riskCode, data.id_found.astype(np.int32),
                                                       data.riskcode_301_level.astype(np.int32))
        feaOfid_foundAndrisk503 = self.getCrossFeature(foundAndIdfound, riskCode, data.id_found.astype(np.int32),
                                                       data.riskcode_503_level.astype(np.int32))

        return np.c_[
            feaOfRisk301AndRisk503, feaOfFoundAndrisk1, feaOfFoundAndrisk2, feaOfFoundAndrisk3, feaOfFoundAndrisk4, feaOfFoundAndrisk5, feaOfFoundAndrisk6, feaOfFoundAndrisk7, feaOfFoundAndrisk8, feaOfFoundAndrisk301, feaOfFoundAndrisk503, feaOfsexAndrisk1, feaOfsexAndrisk2, feaOfsexAndrisk3, feaOfsexAndrisk4, feaOfsexAndrisk5, feaOfsexAndrisk6, feaOfsexAndrisk7, feaOfsexAndrisk8, feaOfsexAndrisk301, feaOfsexAndrisk503, feaOfid_foundAndrisk1, feaOfid_foundAndrisk2, feaOfid_foundAndrisk3, feaOfid_foundAndrisk4, feaOfid_foundAndrisk5, feaOfid_foundAndrisk6, feaOfid_foundAndrisk7, feaOfid_foundAndrisk8, feaOfid_foundAndrisk301, feaOfid_foundAndrisk503]

    def validationWideDataset(self,path):
        data = pd.read_csv(path)
        data = data[self.name]
        data = data.fillna(0)
        label = data.label.as_matrix()
        data = data.drop(columns=['label'])
        crossFeature = self.getCrossFeatures(data)
        data = np.c_[data.as_matrix(), crossFeature]
        return data, label


    def validationDataset(self,path1,path2):
        wideData,label = self.validationWideDataset(path1)
        testDnnData, testRnnData, testLabel = self.hnn.validationDataset(path1, path2)
        return wideData, testDnnData, testRnnData, testLabel

    def creatBatchTrain(self, wideData, dnnData, rnnData, label):
        wideData = np.array(wideData)
        dnnData = np.array(dnnData)
        rnnData = np.array(rnnData)
        label = np.array(label)
        length = dnnData.shape[0]
        num = length // self.para.batchSize
        arr = [i for i in range(1, num)]
        random.shuffle(arr)
        for i in arr:
            yield wideData[(i - 1) * self.para.batchSize: i * self.para.batchSize],\
                  dnnData[(i - 1) * self.para.batchSize: i * self.para.batchSize], \
                  rnnData[(i - 1) * self.para.batchSize: i * self.para.batchSize], \
                  label[(i - 1) * self.para.batchSize:i * self.para.batchSize]

    def writeParam(self, pro):
        name = os.path.join("%s/param.txt" % self.logdir)
        with open(name, "a") as fw:
            fw.write(str(pro) + "\n")

    def train(self,epoch=10):
        self.optimize(self.network())

        with tf.Session() as sess:
            self.tebsorboardInit(sess)
            tf.initialize_all_variables().run()
            testWideData, testDnnData, testRnnData, testLabel1 = self.validationDataset(self.hnn.dnn.testPath,self.hnn.rnn.testPath)
            trainWideData, trainDnnData, trainRnnData, trainLabel = self.validationDataset(self.hnn.dnn.trainPath,
                                                                                       self.hnn.rnn.trainPath)
            testLabel = tf.one_hot(testLabel1, self.para.outputDim).eval(session=sess)
            trainLabel = tf.one_hot(trainLabel, self.para.outputDim).eval(session=sess)
            step=1
            for i in range(epoch):

                for batchWideTrain, batchDnnTrain, batchRnnTrain,batchLabel in self.creatBatchTrain(trainWideData, trainDnnData, trainRnnData, trainLabel):

                    tf.local_variables_initializer().run()
                    auc,loss,_= sess.run([self.auc,self.loss,self.train_op],feed_dict={self.wideInput:batchWideTrain,self.hnn.dnn.input:batchDnnTrain,self.hnn.rnn.input:batchRnnTrain,self.output:batchLabel,self.hnn.keep_prob: 1,self.hnn.dnnProb:1,self.hnn.rnnProb:1})
                    if step%100==0:
                        print("step:%d, train loss:%f, auc:%f" %(step,loss,auc[1]))
                        tf.local_variables_initializer().run()
                        predict, auc = sess.run([self.predict, self.auc],
                                                              feed_dict={self.wideInput:testWideData,self.hnn.dnn.input:testDnnData,self.hnn.rnn.input:testRnnData,self.output:testLabel,self.hnn.keep_prob: 1,self.hnn.dnnProb:1,self.hnn.rnnProb:1})

                        ks = metrics_ks(np.array(testLabel1), predict[:, 1])
                        print("ks value: %f, auc : %f " %(ks,auc[1]))
                        self.writeParam(ks)
                    step += 1

if __name__=="__main__":
    paramDNN = paDNN()
    paramDNN.outputDim=64
    paramRNN = paRNN()
    paramRNN.outputDim=64
    paramHNN = paraHNN()
    paramHNN.inputDim=paramDNN.outputDim+paramRNN.outputDim
    paramHNN.outputDim=128
    param.concatDim = paramHNN.outputDim+param.inputDim
    paramHNN.batchSize =paramRNN.batchSize = paramDNN.batchSize = 32
    testDnnPath = "../water_info/data/3test_full.csv"
    trainDnnPath = "../water_info/data/3train_full.csv"
    testRnnPath = "../water_info/data/3test_ba.csv"
    trainRnnPath = "../water_info/data/3train_ba.csv"
    dictPath = "../water_info/data/subappid.csv"
    weightPath="../water_info/data/w2.pickle"
    test = wideDeep(param,HNN,paramHNN,paramDNN,paramRNN,DNN,RNN,testDnnPath,trainDnnPath,testRnnPath,trainRnnPath,dictPath,weightPath)
    test.train(epoch=10)