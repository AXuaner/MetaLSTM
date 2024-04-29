import pickle
import sys
from torch import nn
sys.path.append('../')
from hydroDL import utils
from hydroDL.master import default,master
from hydroDL.post import plot, stat
import matplotlib.pyplot as plt
from hydroDL.data import camels
from hydroDL.model import rnn, crit, train, meta_train
import numpy as np
import pandas as pd
import os
import torch
import random
import datetime as dt
import json
from matplotlib import rcParams


interfaceOpt = 1
config = {"font.family": 'Times New Roman', "font.size": 12}
rcParams.update(config)
plt.rcParams['font.size'] = 12
# ==1 default, the recommended and more interpretable version with clear data and training flow. We improved the
# original one to explicitly load and process data, set up model and loss, and train the model.
# ==0, the original "pro" version to train jobs based on the defined configuration dictionary.
# Results are very similar for two options.

# Options for training and testing
# 0: train base model
# 1: train meta model
# 0,1: do both base and meta model
# 2: test trained models
Action = [1]
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCH = 300
BATCH_SIZE = 100
RHO = 365
HIDDENSIZE = 256
saveEPOCH = 10 # save model for every "saveEPOCH" epochs
Ttrain = [19851001, 19951001]  # Training period

# Fix random seed
seedid = 888
random.seed(seedid)
torch.manual_seed(seedid)
np.random.seed(seedid)
torch.cuda.manual_seed(seedid)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
current_folder = os.path.dirname(os.path.abspath(__file__))
rootDatabase = os.path.join(current_folder, 'scratch', 'Camels')  # CAMELS dataset root directory: /scratch/Camels
camels.initcamels(rootDatabase)  # initialize three camels module-scope variables in camels.py: dirDB, gageDict, statDict
                                #dirDB指定camel数据集根目录、gage读取camel中的流域信息，如地理位置，气象数据、statdict将Json文件转存进来
rootOut = os.path.join(current_folder, 'data', 'rnnStreamflow')  # Root directory to save training results: /data/rnnStreamflow

# Define all the configurations into dictionary variables
# three purposes using these dictionaries. 1. saved as configuration logging file. 2. for future testing. 3. can also
# be used to directly train the model when interfaceOpt == 0

# define dataset
# default module stores default configurations, using update to change the config
optData = default.optDataCamels
optData = default.update(optData, varT=camels.forcingLst, varC=camels.attrLstSel, tRange=Ttrain)  # Update the training period

if (interfaceOpt == 1) and (4 not in Action):
    # read data from original CAMELS dataset
    # df: CAMELS dataframe; x: forcings[nb,nt,nx]; y: streamflow obs[nb,nt,ny]; c:attributes[nb,nc]
    # nb: number of basins, nt: number of time steps (in Ttrain), nx: number of time-dependent forcing variables
    # ny: number of target variables, nc: number of constant attributes
    df = camels.DataframeCamels(
        subset=optData['subset'], tRange=optData['tRange'])
    x = df.getDataTs(
        varLst=optData['varT'],
        doNorm=False,
        rmNan=False)

    y = df.getDataObs(
        doNorm=False,
        rmNan=False,
        basinnorm=False)

    with open('lstm.pickle', 'rb') as file:
        basin_mask = pickle.load(file)
    basin_mask = np.squeeze(basin_mask)

    y_temp = camels.basinNorm(y, optData['subset'], toNorm=True)
    c = df.getDataConst(
        varLst=optData['varC'],
        doNorm=False,
        rmNan=False)

    # process, do normalization and remove nan
    series_data = np.concatenate([x, y_temp], axis=2)
    seriesvarLst = camels.forcingLst + ['runoff']
    # calculate statistics for norm and saved to a dictionary
    statDict = camels.getStatDic(attrLst=camels.attrLstSel, attrdata=c, seriesLst=seriesvarLst, seriesdata=series_data)
    # normalize
    attr_norm = camels.transNormbyDic(c, camels.attrLstSel, statDict, toNorm=True)
    attr_norm[np.isnan(attr_norm)] = 0.0
    series_norm = camels.transNormbyDic(series_data, seriesvarLst, statDict, toNorm=True)

    # prepare the inputs
    xTrain = series_norm[:, :, :-1]
    xTrain[np.isnan(xTrain)] = 0.0
    yTrain = np.expand_dims(series_norm[:, :, -1], 2)
    attrs = attr_norm

# define model and update configure
if torch.cuda.is_available():
    optModel = default.optLstm
else:
    optModel = default.update(
        default.optLstm,
        name='hydroDL.model.rnn.CpuLstmModel')
optModel = default.update(default.optLstm, hiddenSize=HIDDENSIZE)
optLoss = default.optLossRMSE
optTrain = default.update(default.optTrainCamels, miniBatch=[BATCH_SIZE, RHO], nEpoch=EPOCH, saveEpoch=saveEPOCH, seed=seedid)

# define output folder for model results
exp_name = 'CAMELSDemo'
exp_disp = 'TestRun'
save_path = os.path.join(exp_name, exp_disp, \
            'epochs{}_batch{}_rho{}_hiddensize{}_Tstart{}_Tend{}'.format(optTrain['nEpoch'], optTrain['miniBatch'][0],
                                                                          optTrain['miniBatch'][1],
                                                                          optModel['hiddenSize'],
                                                                          optData['tRange'][0], optData['tRange'][1]))

# Train the base model without data integration
if 0 in Action:
    out = os.path.join(rootOut, save_path, 'LSTM')  # output folder to save results
    # Wrap up all the training configurations to one dictionary in order to save into "out" folder
    masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
    if interfaceOpt == 1:  # use the more interpretable version interface
        nx = xTrain.shape[-1] + attrs.shape[-1]  # update nx, nx = nx + nc
        ny = yTrain.shape[-1]
        # load model for training
        if torch.cuda.is_available():
            model = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
            model = nn.DataParallel(model)
            model.to(device)
        else:
            model = rnn.CpuLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
        optModel = default.update(optModel, nx=nx, ny=ny)
        lossFun = crit.RmseLoss()
        masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
        master.writeMasterFile(masterDict)
        # log statistics
        statFile = os.path.join(out, 'statDict.json')
        with open(statFile, 'w') as fp:
            json.dump(statDict, fp, indent=4)
        # train model
        model = train.trainModel(
            model,
            xTrain,
            yTrain,
            attrs,
            lossFun,
            nEpoch=EPOCH,
            miniBatch=[BATCH_SIZE, RHO],
            saveEpoch=saveEPOCH,
            saveFolder=out)
    elif interfaceOpt==0: # directly train the model using dictionary variable
        master.train(masterDict)

if 1 in Action:
    out = os.path.join(rootOut, save_path, 'Meta')  # output folder to save results
    # Wrap up all the training configurations to one dictionary in order to save into "out" folder
    masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
    if interfaceOpt == 1:  # use the more interpretable version interface
        nx = xTrain.shape[-1] + attrs.shape[-1]  # update nx, nx = nx + nc
        ny = yTrain.shape[-1]
        # load model for trainingNV
        if torch.cuda.is_available():
            model = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
            model = nn.DataParallel(model)
            model.to(device)
        else:
            model = rnn.CpuLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
        optModel = default.update(optModel, nx=nx, ny=ny)
        lossFun = crit.RmseLoss()
        masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
        master.writeMasterFile(masterDict)
        # log statistics
        statFile = os.path.join(out, 'statDict.json')
        with open(statFile, 'w') as fp:
            json.dump(statDict, fp, indent=4)
        # train model
        model = meta_train.meta_trainModel(
            model,
            xTrain,
            yTrain,
            basin_mask,
            attrs,
            lossFun,
            nEpoch=EPOCH,
            miniBatch=[BATCH_SIZE, RHO],
            saveEpoch=saveEPOCH,
            saveFolder=out)
    elif interfaceOpt==0: # directly train the model using dictionary variable
        master.train(masterDict)


# Test models
if 2 in Action:
    device = torch.device("cuda:0")
    # 将模型移动到指定的 GPU
    TestEPOCH = 100 # choose the model to test after trained "TestEPOCH" epoches源代码此处为TestEPOCH = 300
    # generate a folder name list containing all the tested model output folders
    directory = os.path.join(rootOut, save_path)
    #file_names = os.listdir(directory)
    file_names = ['A','Meta-A']
    outLst = [os.path.join(directory, file_name) for file_name in file_names]
   # outLst includes all the directories to test
    subset = 'All'  # 'All': use all the CAMELS gages to test; Or pass the gage list
    tRange = [19951001, 20051001]  # Testing period
    testBatch = 100 # do batch forward to save GPU memory
    predLst = list()
    for out in outLst:
        if interfaceOpt == 1:  # use the more interpretable version interface
            # load testing data
            mDict = master.readMasterFile(out)
            optData = mDict['data']
            df = camels.DataframeCamels(
                subset=subset, tRange=tRange)
            x = df.getDataTs(
                varLst=optData['varT'],
                doNorm=False,
                rmNan=False)
            obs = df.getDataObs(
                doNorm=False,
                rmNan=False,
                basinnorm=False)
            c = df.getDataConst(
                varLst=optData['varC'],
                doNorm=False,
                rmNan=False)

            statFile = os.path.join(out, 'statDict.json')
            with open(statFile, 'r') as fp:
                statDict = json.load(fp)
            seriesvarLst = optData['varT']
            attrLst = optData['varC']
            attr_norm = camels.transNormbyDic(c, attrLst, statDict, toNorm=True)
            attr_norm[np.isnan(attr_norm)] = 0.0
            xTest = camels.transNormbyDic(x, seriesvarLst, statDict, toNorm=True)
            xTest[np.isnan(xTest)] = 0.0
            attrs = attr_norm

            if optData['daObs'] > 0:
                # optData['daObs'] != 0, load previous observation data to integrate
                nDay = optData['daObs']
                sd = utils.time.t2dt(
                    tRange[0]) - dt.timedelta(days=nDay)
                ed = utils.time.t2dt(
                    tRange[1]) - dt.timedelta(days=nDay)
                dfdi = camels.DataframeCamels(
                    subset=subset, tRange=[sd, ed])
                datatemp = dfdi.getDataObs(
                    doNorm=False, rmNan=False, basinnorm=True) # 'basinnorm=True': output = discharge/(area*mean_precip)
                # normalize data
                dadata = camels.transNormbyDic(datatemp, 'runoff', statDict, toNorm=True)
                dadata[np.where(np.isnan(dadata))] = 0.0
                xIn = np.concatenate([xTest, dadata], axis=2)

            else:
                xIn = xTest

            # load and forward the model for testing
            testmodel = master.loadModel(out, epoch=TestEPOCH)
            testmodel = testmodel.to(device)
            filePathLst = master.namePred(
                out, tRange, 'All', epoch=TestEPOCH)  # prepare the name of csv files to save testing results

            train.testModel(
                testmodel, xIn, c=attrs, batchSize=testBatch, filePathLst=filePathLst)
            # read out predictions
            dataPred = np.ndarray([obs.shape[0], obs.shape[1], len(filePathLst)])
            for k in range(len(filePathLst)):
                filePath = filePathLst[k]
                dataPred[:, :, k] = pd.read_csv(
                    filePath, dtype=np.float, header=None).values
            # transform back to the original observation
            temppred = camels.transNormbyDic(dataPred, 'runoff', statDict, toNorm=False)
            pred = camels.basinNorm(temppred, subset, toNorm=False)

        elif interfaceOpt == 0: # only for models trained by the pro interface
            df, pred, obs = master.test(out, tRange=tRange, subset=subset, batchSize=testBatch, basinnorm=True,
                                        epoch=TestEPOCH, reTest=True)

        # change the units ft3/s to m3/s
        obs = obs * 0.0283168
        pred = pred * 0.0283168
        predLst.append(pred) # the prediction list for all the models


    # calculate statistic metrics
    statDictLst = [stat.statError(x.squeeze(), obs.squeeze()) for x in predLst]

    with open('statDictLst.pickle', 'wb') as file:
        pickle.dump(statDictLst, file)

    data = statDictLst[6]['NSE']

    keyLst = ['Bias', 'NSE', 'FLV', 'FHV']
    data = []
    subset = 'All'  # 'All': use all the CAMELS gages to test; Or pass the gage list
    tRange = [19951001, 20051001]  # Testing period
    testBatch = 100  # do batch forward to save GPU memory
    predLst = list()

    # 遍历每个模型的统计数据
    for i, stat in enumerate(statDictLst):
        temp = []
        for key in keyLst:
            data0 = stat[key]
            data0 = data0[~np.isnan(data0)]
            temp.append(data0)
        data.append(temp)
    j = len(data)

    xlabel = ['Bias ($\mathregular{m^3}$/s)', 'KGE', 'FLV(%)', 'FHV(%)']
    # 创建一个 Figure 对象和包含七个子图的 Axes 对象数组
    fig, axs = plt.subplots(1, 4, figsize=(10, 4), sharey=False, sharex=True)
    # 遍历每组数据并绘制子图
    j = len(data)
    ticks = list(range(j))
    colors = plt.cm.Set3.colors
    for i in range(4):
        len1 = 0
        for len1 in range(j):
            # 在当前子图中绘制第一组数据的箱线图
            box = axs[i].boxplot(data[len1][i], 0, '', positions=[len1], widths=0.6, patch_artist=True)
            for patch in box['boxes']:
                patch.set(facecolor=colors[len1])  # 你可以更改颜色
        axs[i].text(-0.4, 0.5, xlabel[i], fontsize=8, rotation=90, transform=axs[i].transAxes, va='center')
        axs[i].set_xticks(ticks)
        axs[i].set_xticklabels(file_names, rotation=90)
        axs[i].tick_params(left=True, labelleft=True)
        axs[i].yaxis.grid(True)
    # 设置整个图的标题和布局
    fig.suptitle('Boxplots of Data', y=1.2)
    fig.savefig('boxplots.png')
    plt.tight_layout()
    plt.show()
