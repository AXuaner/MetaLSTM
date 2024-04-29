import torch
import numpy as np
import math
import torch.nn as nn

def calculate_mean_squared_increment(data, x_rainsign):
    loss0 = 0.0
    cnt = 0
    a = []

    for region in range(data.shape[1]):
        for day in range(1, data.shape[0]):
            if x_rainsign[day, region, 0] == x_rainsign[day-1 , region, 0]:
                d1 = data[day, region, 0].cpu().item()
                d2 = data[day-1 , region, 0].cpu().item()
                difference = (d1 - d2) ** 2
                loss0 = loss0 + difference
                cnt = cnt + 1
            else:
                if cnt > 0:
                    a.append(loss0 / cnt)
                loss0 = 0.0
                cnt = 0

        if cnt > 0:
            a.append(loss0 / cnt)

    if  len(a) > 0:  # 检查列表是否包含 NaN 值并确保列表不为空
        loss = np.nanmean(a)
    else:
        loss = 0.0  # 如果没有有效的数据，可以选择返回0或其他默认值

    return loss

class RmseLoss_infinity(torch.nn.Module):
    def __init__(self):
        super(RmseLoss_infinity, self).__init__()

    # Calculate MSE∞
    def forward(self, output, target):
        loss = 0
        temp = 0
        p0 = output[:, :, 0:1]
        t0 = target[:, :, 0:1]
        marks = target[:, :, 1:2]
        indices = np.where(marks == 1)
        if indices[0].size == 0:
            loss = 0
        else:
            t1 = t0[indices]
            p1 = p0[indices]
            # 计算标记为1的降雨特征的平方
            temp = torch.sqrt(((p1-t1) ** 2).mean())
            loss = loss + temp

        return loss


class RmseLoss_0(torch.nn.Module):
    def __init__(self):
        super(RmseLoss_0, self).__init__()

    # Calculate MSE0
    def forward(self, output, target):
        loss = 0
        temp = 0
        p0 = output[:, :, 0:1]
        marks = target[:, :, 1:2]
        indices = np.where(marks == 2)
        if indices[0].size == 0:
            loss = 0
        else:
            p1 = p0[indices]
            # 计算标记为1的降雨特征的平方
            temp = torch.sqrt(((p1 ) ** 2).mean())
            loss = loss + temp

        return loss


class RmseLoss_mcp(torch.nn.Module):#在train.py第114行
    def __init__(self):
        super(RmseLoss_mcp, self).__init__()

    # Calculate MSE_mcp
    def forward(self, output, results):
        p0 = output[:, :, 0:1]
        t0 = results[:, :, 0:1]
        loss = calculate_mean_squared_increment(p0 , t0)

        return loss

class RmseLoss_mce(torch.nn.Module):
    def __init__(self):
        super(RmseLoss_mce, self).__init__()
    # Calculate MSE_mce
    def forward(self, output, target):
        loss = 0
        return loss

class RmseLoss_sum5(torch.nn.Module):
    def __init__(self):
        super(RmseLoss_sum5, self).__init__()
        # 创建第一个损失函数的实例
        self.rmse_loss = RmseLoss()
        self.rmse_loss_infinity = RmseLoss_infinity()
        self.rmse_loss_0 = RmseLoss_0()
        self.rmse_loss_mcp = RmseLoss_mcp()
        self.rmse_loss_mce = RmseLoss_mce()

    def forward(self, inputs, targets, yy_p, yy_train, rain):
        # 在第二个损失函数中调用第一个损失函数的forward方法
        loss0 = self.rmse_loss(inputs, targets)
        loss1 = self.rmse_loss_infinity(yy_p, yy_train)
        loss2 = self.rmse_loss_0(yy_p, yy_train)
        loss3 = self.rmse_loss_mcp(inputs, rain)
        loss4 = self.rmse_loss_mce(inputs, rain)
        loss = ( 0.6 * loss0 + 0.1 * loss1 + 0.1 * loss2+ 0.1 * loss3 + 0.1 * loss4 )

        return loss

class RmseLoss(torch.nn.Module):
    def __init__(self):
        super(RmseLoss, self).__init__()

    def forward(self, output, target):

        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            temp = torch.sqrt(((p - t)**2).mean())
            loss = loss + temp

        return loss

class SigmaLoss(torch.nn.Module):
    def __init__(self, prior='gauss'):
        super(SigmaLoss, self).__init__()
        self.reduction = 'elementwise_mean'
        if prior == '':
            self.prior = None
        else:
            self.prior = prior.split('+')

    def forward(self, output, target):
        ny = target.shape[-1]
        lossMean = 0
        for k in range(ny):
            p0 = output[:, :, k * 2]
            s0 = output[:, :, k * 2 + 1]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            s = s0[mask]
            t = t0[mask]
            if self.prior[0] == 'gauss':
                loss = torch.exp(-s).mul((p - t)**2) / 2 + s / 2
            elif self.prior[0] == 'invGamma':
                c1 = float(self.prior[1])
                c2 = float(self.prior[2])
                nt = p.shape[0]
                loss = torch.exp(-s).mul(
                    (p - t)**2 + c2 / nt) / 2 + (1 / 2 + c1 / nt) * s
            lossMean = lossMean + torch.mean(loss)
        return lossMean

class RmseLossCNN(torch.nn.Module):
    def __init__(self):
        super(RmseLossCNN, self).__init__()

    def forward(self, output, target):
        # output = ngrid * nvar * ntime
        ny = target.shape[1]
        loss = 0
        for k in range(ny):
            p0 = output[:, k, :]
            t0 = target[:, k, :]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            temp = torch.sqrt(((p - t)**2).mean())
            loss = loss + temp
        return loss

class RmseLossANN(torch.nn.Module):
    def __init__(self, get_length=False):
        super(RmseLossANN, self).__init__()
        self.ind = get_length

    def forward(self, output, target):
        if len(output.shape) == 2:
            p0 = output[:, 0]
            t0 = target[:, 0]
        else:
            p0 = output[:, :, 0]
            t0 = target[:, :, 0]
        mask = t0 == t0
        p = p0[mask]
        t = t0[mask]
        loss = torch.sqrt(((p - t)**2).mean())
        if self.ind is False:
            return loss
        else:
            Nday = p.shape[0]
            return loss, Nday

class ubRmseLoss(torch.nn.Module):
    def __init__(self):
        super(ubRmseLoss, self).__init__()

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            pmean = p.mean()
            tmean = t.mean()
            p_ub = p-pmean
            t_ub = t-tmean
            temp = torch.sqrt(((p_ub - t_ub)**2).mean())
            loss = loss + temp
        return loss

class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, target):
        ny = target.shape[2]
        loss = 0
        for k in range(ny):
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            temp = ((p - t)**2).mean()
            loss = loss + temp
        return loss

class NSELoss(torch.nn.Module):
    def __init__(self):
        super(NSELoss, self).__init__()

    def forward(self, output, target):
        Ngage = target.shape[1]
        losssum = 0
        nsample = 0
        for ii in range(Ngage):
            p0 = output[:, ii, 0]
            t0 = target[:, ii, 0]
            mask = t0 == t0
            if len(mask[mask==True])>0:
                p = p0[mask]
                t = t0[mask]
                tmean = t.mean()
                SST = torch.sum((t - tmean) ** 2)
                if SST != 0:
                    SSRes = torch.sum((t - p) ** 2)
                    temp = 1 - SSRes / SST
                    losssum = losssum + temp
                    nsample = nsample +1
        # minimize the opposite average NSE
        loss = -(losssum/nsample)
        return loss

class NSELosstest(torch.nn.Module):
    # Same as Fredrick 2019
    def __init__(self):
        super(NSELosstest, self).__init__()

    def forward(self, output, target):
        Ngage = target.shape[1]
        losssum = 0
        nsample = 0
        for ii in range(Ngage):
            p0 = output[:, ii, 0]
            t0 = target[:, ii, 0]
            mask = t0 == t0
            if len(mask[mask==True])>0:
                p = p0[mask]
                t = t0[mask]
                tmean = t.mean()
                SST = torch.sum((t - tmean) ** 2)
                SSRes = torch.sum((t - p) ** 2)
                temp = SSRes / ((torch.sqrt(SST)+0.1)**2)
                losssum = losssum + temp
                nsample = nsample +1
        loss = losssum/nsample
        return loss


class TrendLoss(torch.nn.Module):
    # Add the trend part to the loss
    def __init__(self):
        super(TrendLoss, self).__init__()

    def getSlope(self, x):
        idx = 0
        n = len(x)
        d = torch.ones(int(n * (n - 1) / 2))

        for i in range(n - 1):
            j = torch.arange(start=i + 1, end=n)
            d[idx: idx + len(j)] = (x[j] - x[i]) / (j - i).type(torch.float)
            idx = idx + len(j)

        return torch.median(d)

    def forward(self, output, target, PercentLst=[100, 98, 50, 30, 2]):
        # output, target: rho/time * Batchsize * Ntraget_var
        ny = target.shape[2]
        nt = target.shape[0]
        ngage = target.shape[1]
        loss = 0
        for k in range(ny):
        # loop for variable
            p0 = output[:, :, k]
            t0 = target[:, :, k]
            mask = t0 == t0
            p = p0[mask]
            t = t0[mask]
            # first part loss, regular RMSE
            temp = torch.sqrt(((p - t)**2).mean())
            loss = loss + temp
            temptrendloss = 0
            nsample = 0
            for ig in range(ngage):
                # loop for basins
                pgage0 = p0[:, ig].reshape(-1, 365)
                tgage0 = t0[:, ig].reshape(-1, 365)
                gBool = np.zeros(tgage0.shape[0]).astype(int)
                pgageM = torch.zeros(tgage0.shape[0])
                pgageQ = torch.zeros(tgage0.shape[0], len(PercentLst))
                tgageM = torch.zeros(tgage0.shape[0])
                tgageQ = torch.zeros(tgage0.shape[0], len(PercentLst))
                for ii in range(tgage0.shape[0]):
                    pgage = pgage0[ii, :]
                    tgage = tgage0[ii, :]
                    maskg = tgage == tgage
                    # quality control
                    if maskg.sum() > (1-2/12)*365:
                        gBool[ii] = 1
                        pgage = pgage[maskg]
                        tgage = tgage[maskg]
                        pgageM[ii] = pgage.mean()
                        tgageM[ii] = tgage.mean()
                        for ip in range(len(PercentLst)):
                            k = math.ceil(PercentLst[ip] / 100 * 365)
                            # pgageQ[ii, ip] = torch.kthvalue(pgage, k)[0]
                            # tgageQ[ii, ip] = torch.kthvalue(tgage, k)[0]
                            pgageQ[ii, ip] = torch.sort(pgage)[0][k-1]
                            tgageQ[ii, ip] = torch.sort(tgage)[0][k-1]
                # Quality control
                if gBool.sum()>6:
                    nsample = nsample + 1
                    pgageM = pgageM[gBool]
                    tgageM = tgageM[gBool]
                    # mean annual trend loss
                    temptrendloss = temptrendloss + (self.getSlope(tgageM)-self.getSlope(pgageM))**2
                    pgageQ = pgageQ[gBool, :]
                    tgageQ = tgageQ[gBool, :]
                    # quantile trend loss
                    for ii in range(tgageQ.shape[1]):
                        temptrendloss = temptrendloss + (self.getSlope(tgageQ[:, ii])-self.getSlope(pgageQ[:, ii]))**2

            loss = loss + temptrendloss/nsample

        return loss


class ModifyTrend(torch.nn.Module):
    # Add the trend part to the loss
    def __init__(self):
        super(ModifyTrend, self).__init__()

    def getSlope(self, x):
        nyear, ngage = x.shape
        # define difference matirx
        x = x.transpose(0,1)
        xtemp = x.repeat(1, nyear)
        xi = xtemp.reshape([ngage, nyear, nyear])
        xj = xi.transpose(1,2)
        # define i,j matrix
        im = torch.arange(nyear).repeat(nyear).reshape(nyear,nyear).type(torch.float)
        im = im.unsqueeze(0).repeat([ngage, 1, 1])
        jm = im.transpose(1,2)
        delta = 1.0/(im - jm)
        delta = delta.cuda()
        # calculate the slope matrix
        slopeMat = (xi - xj)*delta
        rid, cid = np.triu_indices(nyear, k=1)
        slope = slopeMat[:, rid, cid]
        senslope = torch.median(slope, dim=-1)[0]

        return senslope


    def forward(self, output, target, PercentLst=[-1]):
        # output, target: rho/time * Batchsize * Ntraget_var
        # PercentLst = [100, 98, 50, 30, 2, -1]
        ny = target.shape[2]
        nt = target.shape[0]
        ngage = target.shape[1]
        # loop for variable
        p0 = output[:, :, 0]
        t0 = target[:, :, 0]
        mask = t0 == t0
        p = p0[mask]
        t = t0[mask]
        # first part loss, regular RMSE
        # loss = torch.sqrt(((p - t)**2).mean())
        # loss = ((p - t) ** 2).mean()
        loss = 0
        temptrendloss = 0
        # second loss: adding trend
        p1 = p0.reshape(-1, 365, ngage)
        t1 = t0.reshape(-1, 365, ngage)
        for ip in range(len(PercentLst)):
            k = math.ceil(PercentLst[ip] / 100 * 365)
            # pQ = torch.kthvalue(p1, k, dim=1)[0]
            # tQ = torch.kthvalue(t1, k, dim=1)[0]
            # output: dim=Year*gage
            if PercentLst[ip]<0:
                pQ = torch.mean(p1, dim=1)
                tQ = torch.mean(t1, dim=1)
            else:
                pQ = torch.sort(p1, dim=1)[0][:, k - 1, :]
                tQ = torch.sort(t1, dim=1)[0][:, k - 1, :]
            # temptrendloss = temptrendloss + ((self.getSlope(pQ) - self.getSlope(tQ)) ** 2).mean()
            temptrendloss = temptrendloss + ((pQ - tQ) ** 2).mean()
        loss = loss + temptrendloss

        return loss


class ModifyTrend1(torch.nn.Module):
    # Add the trend part to the loss
    def __init__(self):
        super(ModifyTrend1, self).__init__()

    def getM(self, n):
        M = np.zeros([n**2, n])
        s0 = np.zeros([n**2, 1])
        for j in range (n):
            for i in range(n):
                k = j*n+i
                if i<j:
                    factor = 1/(j-i)
                    M[k, j] = factor
                    M[k, i] = -factor
                else:
                    s0[k] = np.nan
        sind = np.argwhere(~np.isnan(s0))
        return M, sind


    def forward(self, output, target, PercentLst=[100, 98, 50, 30, 2]):
        # PercentLst = [100, 98, 50, 30, 2]
        # output, target: rho/time * Batchsize * Ntraget_var
        output = output.permute(2, 0, 1)
        target = target.permute(2, 0, 1)
        ny = target.shape[2]
        nt = target.shape[0]
        ngage = target.shape[1]
        # loop for variable
        p0 = output[:, :, 0]
        t0 = target[:, :, 0]
        # mask = t0 == t0
        # p = p0[mask]
        # t = t0[mask]
        # first part loss, regular RMSE
        loss = 0.0
        # loss = torch.sqrt(((p - t)**2).mean())
        # loss = ((p - t) ** 2).mean()
        # second loss: adding trend
        p1 = p0.reshape(-1, 365, ngage)
        t1 = t0.reshape(-1, 365, ngage)
        nyear = p1.shape[0]
        nsample = p1.shape[-1]
        M, s0 = self.getM(nyear)
        Mtensor = torch.from_numpy(M).type(torch.float).cuda()
        for ip in range(len(PercentLst)):
            k = math.ceil(PercentLst[ip] / 100 * 365)
            # pQ = torch.kthvalue(p1, k, dim=1)[0]
            # tQ = torch.kthvalue(t1, k, dim=1)[0]
            # output: dim=Year*gage
            pQ = torch.sort(p1, dim=1)[0][:, k - 1, :]
            tQ = torch.sort(t1, dim=1)[0][:, k - 1, :]
            # pQ = p1[:, 100, :]
            # tQ = t1[:, 100, :]
            temptrenddiff = 0.0
            for ig in range(nsample):
                trenddiff = (torch.median(torch.mv(Mtensor, pQ[:, ig])[s0[:,0]]) - \
                            torch.median(torch.mv(Mtensor, tQ[:, ig])[s0[:,0]]))**2
                temptrenddiff = temptrenddiff + trenddiff
            temptrendloss = temptrenddiff/nsample
            loss = loss + temptrendloss
        return loss

# class ModifyTrend1(torch.nn.Module):
#     # Test MSE loss for each percentile
#     def __init__(self):
#         super(ModifyTrend1, self).__init__()
#
#     def getM(self, n):
#         M = np.zeros([n**2, n])
#         s0 = np.zeros([n**2, 1])
#         for j in range (n):
#             for i in range(n):
#                 k = j*n+i
#                 if i<j:
#                     factor = 1/(j-i)
#                     M[k, j] = factor
#                     M[k, i] = -factor
#                 else:
#                     s0[k] = np.nan
#         sind = np.argwhere(~np.isnan(s0))
#         return M, sind
#
#
#     def forward(self, output, target, PercentLst=[100, 98, 50, 30, 2]):
#         # PercentLst = [100, 98, 50, 30, 2]
#         # output, target: rho/time * Batchsize * Ntraget_var
#         output = output.permute(2, 0, 1)
#         target = target.permute(2, 0, 1)
#         ny = target.shape[2]
#         nt = target.shape[0]
#         ngage = target.shape[1]
#         # loop for variable
#         p0 = output[:, :, 0]
#         t0 = target[:, :, 0]
#         mask = t0 == t0
#         p = p0[mask]
#         t = t0[mask]
#         # first part loss, regular RMSE
#         # loss = 0.0
#         # loss = torch.sqrt(((p - t)**2).mean())
#         loss = ((p - t) ** 2).mean()
#         # second loss: adding trend
#         p1 = p0.reshape(-1, 365, ngage)
#         t1 = t0.reshape(-1, 365, ngage)
#         nyear = p1.shape[0]
#         nsample = p1.shape[-1]
#         M, s0 = self.getM(nyear)
#         Mtensor = torch.from_numpy(M).type(torch.float).cuda()
#         for ip in range(len(PercentLst)):
#             k = math.ceil(PercentLst[ip] / 100 * 365)
#             # output: dim=Year*gage
#             pQ = torch.sort(p1, dim=1)[0][:, k - 1, :]
#             tQ = torch.sort(t1, dim=1)[0][:, k - 1, :]
#             # calculate mse of each percentile flow
#             mask = tQ == tQ
#             ptemp = pQ[mask]
#             ttemp = tQ[mask]
#             temploss = ((ptemp - ttemp) ** 2).mean()
#             loss = loss + temploss
#         return loss