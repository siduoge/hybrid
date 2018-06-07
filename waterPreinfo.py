import pandas as pd
import numpy as np
import random
import time
class preWaterInfo():
    def __init__(self,path,num=10,frac=0.7,balance=0.1):
        self.path = path
        data = pd.read_csv(path)
        self.data = data[['teg_evil_result_level', 'teg_fake_info', 'teg_financial_black', 'teg_criminal', 'cdg_qq_black',
                     'cdg_wx_black', 'cdg_other_black', 'cdg_id_auth', 'cdg_gf_black', 'mig_2008_0_evil_score',
                     'cdg_credit_risk_score', 'mig_2006_evil_score', 'query_date', 'subappid','id_number','phone_number','found','id_found',
                          'riskcode_1_level', 'riskcode_2_level', 'riskcode_3_level', 'riskcode_4_level', 'riskcode_5_level',
                          'riskcode_6_level', 'riskcode_7_level', 'riskcode_8_level', 'riskcode_301_level', 'riskcode_503_level','label']]
        self.num=num
        self.frac=frac
        self.balance=balance
        del data


    def extractAndPad(self):
        print("balance the data")
        start = time.time()
        self.getbalance()
        balance = time.time()
        print("balance time: %f" %(balance-start))

        print("the water information of each client sort by date")
        self.sortbydateEachClient()
        sortBycli = time.time()
        print("sortBycli time: %f" % (sortBycli - balance))

        print("get new feature from id_number")
        self.getNewFeature()
        getNew = time.time()
        print("getNew time: %f" % (getNew-sortBycli))

        print("get the loan times")
        self.loanTimes()
        getLoanTimes=time.time()
        print("getLoan time: %f" % (getLoanTimes-getNew))
        print("train need the same length")
        self.padLength()
        sameLength = time.time()
        print("sameLength time: %f" % (sameLength-getNew ))

        print("get the train and test dataset")
        return self.getTrainAndTest()

    def sortbydateEachClient(self):
        gb = self.data.groupby(by=['phone_number'], sort=False)
        self.data = gb.apply(lambda x: x.sort_values(["query_date"], ascending=True)).reset_index(drop=True)

    def padLength(self):
        def normal(x):
            row = x.shape[0]
            col = x.shape[1]
            if row < self.num:
                temp = np.zeros((self.num - row, col))
                df = pd.DataFrame(data=temp, columns=x.columns.values)
                x = x.reset_index(drop=True)
                df['id_number']=df['id_number'].astype(str)
                for i in range(self.num - row):
                    df.at[i,'phone_number'] = x.at[0,'phone_number']
                    df.at[i,'id_number'] = x.at[0,'id_number']
                    df.at[i,'label']= x.at[0,'label']
                    df.at[i, 'times'] = x.at[0, 'times']
                data = pd.concat([df, x], axis=0)
                return data
            else:
                data = x.iloc[-self.num:]

                return data

        group = self.data.groupby(by=['phone_number'])
        self.data = group.apply(lambda x: normal(x)).reset_index(drop=True)

    def loanTimes(self):
        def loan(x):
            query =x['query_date']
            shape = x.shape[0]
            last = query.as_matrix()[-1]-200
            times = np.sum(query>last)
            x['times']=np.array([times]*shape)
            x['all']=np.array([shape]*shape)
            subid = x['subappid'].astype(np.int64).as_matrix()
            num = len(set(subid))
            x['uniqueAllTimes'] = np.array([num] * shape)
            uniSubidLast = subid[query>last]
            uniNum = len(set(uniSubidLast))
            x['uniqueLastNum'] = np.array([uniNum] * shape)
            return x
        group = self.data.groupby(by=['phone_number'])
        self.data = group.apply(lambda x: loan(x)).reset_index(drop=True)
    def getbalance(self):
        def balance(x):
            x = x.reset_index(drop=True)
            if x.at[0,'label'] == 0:
                if random.randint(101, 200) > int((200 - self.balance * 100)):
                    return x
            else:
                return x

        group = self.data.groupby(by=['phone_number'])
        self.data = group.apply(lambda x: balance(x)).reset_index(drop=True)



    def getNewFeature(self):
        def ageAndSex(id_number):
            if pd.isna(id_number):
                # return (None,None)
                return (0, 0)
            else:
                id_number = str(id_number)
                age = id_number[6:10]
                sex = 0 if int(id_number[-2]) % 2 == 0 else 1
            return [age, sex]

        self.data['age'] = self.data['id_number'].apply(lambda x: ageAndSex(x)[0])
        self.data['sex'] = self.data['id_number'].apply(lambda x: ageAndSex(x)[1])

    def getTrainAndTest(self):

        def TrueLabel(x):
            x = x.reset_index(drop=True)
            if x.at[0,'label'] == 0:
                    return x

        def FalseLabel(x):
            x = x.reset_index(drop=True)
            if x.at[0,'label'] == 1:
                return x
        group = self.data.groupby(by=['phone_number'])
        trueLabel = group.apply(lambda x: TrueLabel(x)).reset_index(drop=True)
        falseLabel = group.apply(lambda x: FalseLabel(x)).reset_index(drop=True)
        trueNum = trueLabel.shape[0] // self.num
        trueNum = int(np.round(self.frac * trueNum))
        falseNum = falseLabel.shape[0]//self.num
        falseNum = int(np.round(self.frac * falseNum))

        trainTrue = trueLabel.iloc[:trueNum * self.num]
        trainFalse = falseLabel.iloc[:falseNum * self.num]
        testTrue = trueLabel.iloc[trueNum * self.num:]
        testFalse = falseLabel.iloc[falseNum * self.num:]
        Train = pd.concat([trainTrue,trainFalse],axis=0)
        Test = pd.concat([testTrue, testFalse], axis=0)
        self.Train = Train.drop(columns=['id_number','phone_number'])
        self.Test = Test.drop(columns=['id_number', 'phone_number'])
        return self.Train,self.Test

    def fullConnect(self):
        num = self.Train.shape[0]
        flag = np.array([True if (i + 1) % self.num == 0 else False for i in range(0, num)])
        train =self.Train[flag]

        num = self.Test.shape[0]
        flag = np.array([True if (i + 1) % self.num == 0 else False for i in range(0, num)])
        test = self.Test[flag]
        return train,test

if __name__=="__main__":
    path = "./data/water_info.csv"
    process = preWaterInfo(path,num=5,frac=0.7,balance=1)
    train,test = process.extractAndPad()
    train.to_csv("./data/3train_ba.csv",index=False)
    test.to_csv("./data/3test_ba.csv",index=False)
    fullTrain,fullTest = process.fullConnect()
    fullTrain.to_csv("data/3train_full.csv", index=False)
    fullTest.to_csv("data/3test_full.csv", index=False)





