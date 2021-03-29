# -*- coding: utf8

import gzip
import json
import numpy as np



def car(A, B):
    C=list()
    for i in A:
        for j in B:
            C.append((i,j))
    return set(C)

def normalize(array):
  if type(array) == type(dict()):
    soma = sum(array.values())
    for i in array.keys():
      array[i] = array[i]/soma
    return array
  return np.array(array)/np.sum(array)


def surprise(pMs, data, hipoteses, tick_range = None, *args):
    if tick_range == None:
        tick_range = range(data.shape[0])
  
    
    # Initiating some variebles:
    # Models distribution
    pMs = np.array(pMs) 
    # Surprise model
    surpriseData = np.zeros(data.shape)
    # pMs list -> to store the pMs of all the iterations
    lista_pM = [pMs]
    # pDMs - probability of data given the models
    pDMs = np.zeros(pMs.shape)
    # pMDs - probability of models given the data
    pMDs = np.zeros(pMs.shape)
  
    diferencas = np.zeros(pMs.shape)
    soma_das_diferencas = np.zeros(pMs.shape)
  
  
    for tick in tick_range:
        # difference between prior and posterior 
        soma_das_diferencas = np.zeros(pMs.shape)
    
        # models x artist
        matriz_diferencas = np.zeros((len(pMs),data.shape[1]))
        
        # Estimating the pMD
        for i in range(len(hipoteses)):
            matriz_diferencas[i] = normalize(data[tick, :]) - normalize(hipoteses[i](data, tick, args))
    
    
        for frequence in range(data.shape[1]): 
            #Para cada crença
            for i in range(len(diferencas)):
                diferencas[i] = matriz_diferencas[i,frequence]
                pDMs[i] = 1 - np.abs(diferencas[i])
            pMDs = pMs*pDMs
    
    
        #At this point we already have pMDs and pMs, lets calculate their divergence
          
            kl = 0
            voteSum = 0
            for j in range(len(pMDs)):
                kl = kl + pMDs[j] * np.log( pMDs[j] / pMs[j])
                voteSum = voteSum + diferencas[j] * pMs[j]
                soma_das_diferencas[j] = soma_das_diferencas[j] + np.abs(diferencas[j])
    
    
            if voteSum >= 0 :
                surpriseData[tick, frequence] = np.abs(kl) 
            else:
                surpriseData[tick, frequence] = -1*np.abs(kl)
          
        #Now lets globally update our model belief.
        
        for j in range(len(pMs)):
            pDMs[j] = 1 - (0.5 * soma_das_diferencas[j])
            pMDs[j] = pMs[j] * pDMs[j]
            pMs[j] = pMDs[j]
    
        # Normalizing the  beliefs
        pMs = normalize(pMs)
    
        lista_pM.append(pMs)
    return surpriseData, lista_pM
