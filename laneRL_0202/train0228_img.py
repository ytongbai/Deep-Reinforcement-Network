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

# Output
outputModelPath = "../Lane_models" 
testoutput = open(para.testlogName, 'w+')
logging.basicConfig(filename=para.logName,level=logging.DEBUG)
logging.info(strftime("%Y-%m-%d %H:%M:%S", gmtime()))


all_para_text = open('para.py').read()
all_model_text = open('model_utils.py').read()
print all_para_text
logging.info(all_para_text)
logging.info(all_model_text)

epochs = para.epochs
epsilon = para.epsilon
# epsilon = 0.1
img_batch_size_ratio = para.img_batch_size_ratio
replay_ind = para.buffer_experience_replay - 1
replay_landmark_fea_np = []
replay_state = []
replay_action = []
replay_reward = []
replay_new_state = []

# CNN model
seleted_model = para.seleted_model
learning_rate = para.learning_rate

#meanData
meanImg = para.meanImg

start_id = 0
# CNN model
#selected_modelPath = para.modelPath + '/' + para.modelName
selected_modelPath = '0'
if seleted_model==2:
    model = get_concat_conv_q_network("0")
elif seleted_model==3:
    model = get_img_conv_q_network(selected_modelPath,para.conv_layer,para.dp)
    if selected_modelPath!='0':
        start_id = int(para.modelName.split('.h5')[0].split('_')[-1])


train_state = 0
test_flag = 1
epochs_id = 1



if test_flag ==1:
    print '===Loading Testing Data==='
    testingSetList = glob.glob(os.path.join(para.database,'test*'+ para.subDataKey))
    clses_t,feas_t,imgs_t,gtes_t = get_features(testingSetList)
    index_test = range(len(testingSetList))

trainingSetList = glob.glob(os.path.join(para.database,'train*'+ para.subDataKey))
for i_epoch in range(start_id + epochs_id, epochs_id + start_id + epochs):
    for i_img_minibatch in range(img_batch_size_ratio+1):
        img_minibatch = random.sample(trainingSetList, len(trainingSetList)/img_batch_size_ratio)
        print '===Loading Training Data==='
        clses,feas,imgs,gtes = get_features(img_minibatch)        
        for j in range(len(img_minibatch)):
            fea = feas[j]                
            lane_class = clses[j]
            groundtruth = gtes[j]
            img = imgs[j]
            img = misc.imresize(img,[100,100,3])

            img = img - meanImg
            initMark_X = []# init values on X aixs
            if lane_class== '1':
                initMark_X = [91.0, 71.0, 51.0, 31.0, 11.0]
            else:
                initMark_X = [11.0, 31.0, 51.0, 71.0, 91.0]
            featureData = [fea, initMark_X, groundtruth, img]

            replay = [replay_landmark_fea_np, replay_state, replay_action, replay_reward, replay_new_state]
            replay_ind, train_state, repaly = \
                    generateExpReplay(model, seleted_model, epsilon, replay, replay_ind, train_state, featureData, i_epoch)
                                          
            if train_state == 1:
                replay_landmark_fea_np, replay_state, replay_y= generateCNNTrainData(model,repaly)
                # pdb.set_trace()
                hist = model.fit([replay_landmark_fea_np,replay_state], [replay_y], batch_size=para.batch_size,\
                     epochs=para.epochs_in_memeory,validation_split=0.1)                    
                
                for loss in hist.history['loss']:
                    str_loss = '===Epoch:{:d}-Img:{:d}-Loss:{:.6f}==='.format(i_epoch,j,loss) 
                    logging.info(str_loss)

            # evaluation
            if train_state == 1 and test_flag==1:
                index_test_minibatch = random.sample(index_test, len(index_test)/img_batch_size_ratio)
                mini_clses_t = [clses_t[i_clses_t] for i_clses_t in index_test_minibatch]
                mini_feas_t = [feas_t[i_clses_t] for i_clses_t in index_test_minibatch]
                mini_imgs_t = [imgs_t[i_clses_t] for i_clses_t in index_test_minibatch]
                mini_gtes_t = [gtes_t[i_clses_t] for i_clses_t in index_test_minibatch]

                hit_cnt, detect_hit_cnt, sup_cnt, test_cnt, avg_step_of_agent = \
                                        evaluate.eval(model, testoutput, \
                                                      [mini_clses_t, mini_feas_t, mini_imgs_t, mini_gtes_t], \
                                                       para, \
                                                       '\n\n===Epoch:{:d}===\n\n'.format(i_epoch))
                reStr = 'Test RL:{:.3f}\nTest Det:{:.3f}\nRL is superior to Det:{:.2f}\nAvg steps:{:.3f}\n'\
                        .format(float(hit_cnt)/test_cnt, float(detect_hit_cnt)/test_cnt, \
                                float(sup_cnt)/test_cnt,avg_step_of_agent)
                print reStr
                logging.info(reStr)                 

    if epsilon > 0.1:
        epsilon -= 0.05  
    if i_epoch%5 == 0:
        string = outputModelPath + '/' + para.keyAttr + '_epoch_' + str(i_epoch) + '.h5'
        model.save_weights(string, overwrite=True)
testoutput.close()