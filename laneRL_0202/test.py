import os
import numpy as np
import random
import argparse
from scipy import misc
import pdb
import glob
from data_utils import get_features,generateCNNTrainData,generateExpReplay
from model_utils import get_concat_conv_q_network,get_img_conv_q_network
import evaluate
import para
import logging
from time import strftime,gmtime

# CNN model
selected_modelPath = para.modelPath + '/' + para.modelName
seleted_model = para.seleted_model

if seleted_model==2:
    model = get_concat_conv_q_network("0")
elif seleted_model==3:
    model = get_img_conv_q_network(selected_modelPath, para.conv_layer, para.dp)

print '===Loading Testing Data==='
testingSetList = glob.glob(os.path.join(para.database,'test*'+ para.subDataKey))
clses_t,feas_t,imgs_t,gtes_t = get_features(testingSetList)

acc_hit, acc_detect, acc_sup, avg_step, reStr = \
                        evaluate.eval0504(model, 0, \
                                      [clses_t, feas_t, imgs_t, gtes_t], \
                                      testingSetList,\
                                       '\n\n===TEST===\n\n')
print reStr 
testoutput = open(para.outputVisPath + '/results.txt', 'w+')
testoutput.writelines(reStr)
testoutput.close()              