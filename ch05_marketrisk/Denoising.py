import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def mp_pdf(var, q, pts): # q=T/N, pts는 일종의 정밀도
    eMin, eMax=var*(1-(1./q)**.5)**2,var*(1+(1./q)**.5)**2
    eVal=np.linspace(eMin, eMax, pts).flatten()
    pdf=q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5 # MP pdf based on eigenvalue space
    pdf=pd.Series(pdf,index=eVal)
    return pdf

def kde_fit(obs, bandwidth, x=None):
    
    if len(obs.shape) == 1:
        obs=obs.reshape(-1,1)
    kde_fit=KernelDensity(bandwidth=bandwidth, kernel='gaussian').fit(obs)
    if x is None:
        x=np.unique(obs).reshape(-1, 1)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    logprob = kde_fit.score_samples(x)
    pdf_kde = pd.Series(np.exp(logprob), index=x.flatten())
    return pdf_kde

def errPDFs(var, eVal, q, bWidth, pts=1000):
    # Fit error
    pdf0=mp_pdf(var, q, pts)  # 이론적 pdf
    pdf1=kde_fit(eVal, bWidth, x=pdf0.index.values)  # 경험적 pdf
    sse=np.sum((pdf1-pdf0)**2)
    return sse

def findMaxEval(eVal, q, bWidth):
    out=minimize(lambda *x:errPDFs(*x), .5, args=(eVal, q, bWidth), bounds=((1E-5, 1-1E-5), ))
    if out['success']:
        var=out['x'][0]
    else:
        var=1
    eMax=var*(1+(1./q)**.5)**2
    return eMax, var

def getPCA(matrix):
    eVal, eVec=np.linalg.eigh(matrix)
    indices=eVal.argsort()[::-1]
    eVal, eVec=eVal[indices],eVec[:,indices]
    eVal=np.diagflat(eVal)
    return eVal, eVec

def denoisedCorr(eVal, eVec, nFacts):
    eVal_=np.diag(eVal).copy()
    eVal_[nFacts:]=eVal_[nFacts:].sum()/float(eVal_.shape[0]-nFacts)
    eVal_=np.diag(eVal_)
    cov1=np.dot(eVec, eVal_).dot(eVec.T)
    return cov1

def correlation_to_covariance(corr, std):
    cov = corr * np.outer(std, std)
    return cov

