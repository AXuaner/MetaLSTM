import numpy as np
import torch
import time
import os
from hydroDL.model import rnn
import learn2learn as l2l
from data_gen import _grad_step

def meta_trainModel(model,
               x,
               y,
               basin_mask,
               c,
               lossFun,
               *,
               nEpoch=500,
               miniBatch=[100, 30],
               saveEpoch=100,
               saveFolder=None,
               mode='seq2seq',
               bufftime=0):
    batchSize, rho = miniBatch
    if type(x) is tuple or type(x) is list:
        x, z = x
    ngrid, nt, nx = x.shape
    if c is not None:
        nx = nx + c.shape[-1]
    if batchSize >= ngrid:
        batchSize = ngrid

    nIterEp = int(
        np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid / (nt-bufftime))))
    if hasattr(model, 'ctRm'):
        if model.ctRm is True:
            nIterEp = int(
                np.ceil(
                    np.log(0.01) / np.log(1 - batchSize *
                                          (rho - model.ct) / ngrid / (nt-bufftime))))

    if torch.cuda.is_available():
        lossFun = lossFun.cuda()
        model = model.cuda()

    #定义base learner 和 update learner的优化器
    maml = l2l.algorithms.MAML(model, lr=0.5, first_order=False)
    optim = torch.optim.Adadelta(model.parameters())
    optim2 = torch.optim.Adadelta(maml.parameters(), lr=0.5)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim2, T_max=nEpoch)
    optim.zero_grad()
    optim2.zero_grad()
    #sorted_indices = np.argsort(basin_mask[:,0])
    sorted_indices = np.argsort(basin_mask)
    base_grid = sorted_indices[:-150]
    base_total = base_grid.size
    update_grid = sorted_indices[-150:]
    update_total = update_grid.size
    if saveFolder is not None:
        runFile = os.path.join(saveFolder, 'run.csv')
        rf = open(runFile, 'w+')

    for iEpoch in range(1, nEpoch + 1):
        lossEp = 0
        target_train_error = 0
        lossUpdate = 0
        t0 = time.time()

        for iIter in range(0, nIterEp-15):
            # training iterations
            if type(model.module) in [rnn.CudnnLstmModel, rnn.AnnModel, rnn.CpuLstmModel]:
                iGrid, iT = randomIndex_base(ngrid, nt, base_grid,base_total, [batchSize, rho])
                xTrain = selectSubset(x, iGrid, iT, rho, c=c)
                yTrain = selectSubset(y, iGrid, iT, rho)
                yP = model(xTrain)
            else:
                Exception('unknown model')

            loss = lossFun(yP, yTrain)
            loss.backward()
            optim.step()
            optim.zero_grad()
            lossEp = lossEp + loss.item()
        del xTrain, yTrain, yP
        for iIter in range(0, 15):
            # training iterations
            if type(model.module) in [rnn.CudnnLstmModel, rnn.AnnModel, rnn.CpuLstmModel]:
                # iGrid, iT = randomIndex(ngrid, nt, [batchSize, rho])
                iGrid, iT = randomIndex_base(ngrid, nt, update_grid, update_total, [batchSize, rho])
                xTrain = selectSubset(x, iGrid, iT, rho, c=c)
                # xTrain = rho/time * Batchsize * Ninput_var
                yTrain = selectSubset(y, iGrid, iT, rho)
                # yTrain = rho/time * Batchsize * Ntraget_var
                yP = model(xTrain)
                loss = lossFun(yP, yTrain)
                target_train_error += loss
                torch.cuda.empty_cache()
                #loss.backward()
            # Parameter outer-loop update
        optim2.zero_grad()
        target_train_error.backward()
        _grad_step(maml, iIter, optim2, schedule)
        lossUpdate = lossUpdate + target_train_error.item()
            # if iIter % 30 == 0:
            #     print('Iter {} of {}: Loss {:.3f}'.format(iIter, nIterEp, loss.item()))
        # print loss
        lossTotal = (lossEp+lossUpdate) / (nIterEp*2)
        logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(
            iEpoch, lossTotal,
            time.time() - t0)
        print(logStr)
        # save model and loss
        if saveFolder is not None:
            rf.write(logStr + '\n')
            if iEpoch % saveEpoch == 0:
                # save model
                modelFile = os.path.join(saveFolder,
                                         'model_Ep' + str(iEpoch) + '.pt')
                torch.save(maml, modelFile)
    if saveFolder is not None:
        rf.close()
    return maml


def saveModel(outFolder, model, epoch, modelName='model'):
    modelFile = os.path.join(outFolder, modelName + '_Ep' + str(epoch) + '.pt')
    torch.save(model, modelFile)


def loadModel(outFolder, epoch, modelName='model'):
    modelFile = os.path.join(outFolder, modelName + '_Ep' + str(epoch) + '.pt')
    model = torch.load(modelFile)
    return model


def randomSubset(x, y, dimSubset):
    ngrid, nt, nx = x.shape
    batchSize, rho = dimSubset
    xTensor = torch.zeros([rho, batchSize, x.shape[-1]], requires_grad=False)
    yTensor = torch.zeros([rho, batchSize, y.shape[-1]], requires_grad=False)
    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0, nt - rho, [batchSize])
    for k in range(batchSize):
        temp = x[iGrid[k]:iGrid[k] + 1, np.arange(iT[k], iT[k] + rho), :]
        xTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
        temp = y[iGrid[k]:iGrid[k] + 1, np.arange(iT[k], iT[k] + rho), :]
        yTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    if torch.cuda.is_available():
        xTensor = xTensor.cuda()
        yTensor = yTensor.cuda()
    return xTensor, yTensor


def randomIndex(ngrid, nt, dimSubset, bufftime=0):
    batchSize, rho = dimSubset
    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0+bufftime, nt - rho, [batchSize])
    return iGrid, iT
def randomIndex_base(ngrid, nt, base_grid,base_total, dimSubset, bufftime=0):
    batchSize, rho = dimSubset
    inum = np.random.randint(0, base_total, [batchSize])
    iGrid = base_grid[inum]
    iT = np.random.randint(0+bufftime, nt - rho, [batchSize])
    return iGrid, iT

def selectSubset(x, iGrid, iT, rho, *, c=None, tupleOut=False, LCopt=False, bufftime=0):
    nx = x.shape[-1]
    nt = x.shape[1]
    if x.shape[0] == len(iGrid):   #hack
        iGrid = np.arange(0,len(iGrid))  # hack
    if (rho is not None) and (nt <= rho):
        iT.fill(0)

    batchSize = iGrid.shape[0]
    if iT is not None:
        # batchSize = iGrid.shape[0]
        xTensor = torch.zeros([rho+bufftime, batchSize, nx], requires_grad=False)
        for k in range(batchSize):
            temp = x[iGrid[k]:iGrid[k] + 1, np.arange(iT[k]-bufftime, iT[k] + rho), :]
            xTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    else:
        if LCopt is True:
            # used for local calibration kernel: FDC, SMAP...
            if len(x.shape) == 2:
                # Used for local calibration kernel as FDC
                # x = Ngrid * Ntime
                xTensor = torch.from_numpy(x[iGrid, :]).float()
            elif len(x.shape) == 3:
                # used for LC-SMAP x=Ngrid*Ntime*Nvar
                xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 2)).float()
        else:
            # Used for rho equal to the whole length of time series
            xTensor = torch.from_numpy(np.swapaxes(x[iGrid, :, :], 1, 0)).float()
            rho = xTensor.shape[0]
    if c is not None:
        nc = c.shape[-1]
        temp = np.repeat(
            np.reshape(c[iGrid, :], [batchSize, 1, nc]), rho+bufftime, axis=1)
        cTensor = torch.from_numpy(np.swapaxes(temp, 1, 0)).float()

        if (tupleOut):
            if torch.cuda.is_available():
                xTensor = xTensor.cuda()
                cTensor = cTensor.cuda()
            out = (xTensor, cTensor)
        else:
            out = torch.cat((xTensor, cTensor), 2)
    else:
        out = xTensor

    if torch.cuda.is_available() and type(out) is not tuple:
        out = out.cuda()
    return out