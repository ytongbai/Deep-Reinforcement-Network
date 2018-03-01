import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import argparse
from scipy import ndimage,misc
import pdb
from data_utils import get_state
from reward import update_history_vector
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
import cv2

import para

num_of_steps = para.num_of_steps
seleted_model = para.seleted_model
num_of_landmarks = para.num_of_landmarks
len_of_history_vector = para.num_of_actions*para.num_of_history
ActionStr = para.ActionStr
outputVisPath = para.outputVisPath
margin = 130

def draw_single_action(canvas, i_landmark, i_step, initMark_X, initMark_Y, gt, img):    
    img_start = (margin*i_step+50,margin*(num_of_landmarks-i_landmark)+50)
    canvas[img_start[1]:img_start[1]+100,img_start[0]:img_start[0]+100,:] = img
    for i_gt in range(0,num_of_landmarks):
        # draw GT
        if gt[i_gt]!=-20:
            gt_pt_in_canvas = (img_start[0]+int(gt[i_gt]),img_start[1]+int(initMark_Y[i_gt]))
            cv2.circle(canvas,gt_pt_in_canvas,8,(255,0,0),2)
        # draw init
        if initMark_X[i_gt]!=-20:# skip the removed landmark
            init_pt_in_canvas = (img_start[0]+int(initMark_X[i_gt]),img_start[1]+int(initMark_Y[i_gt]))
            if (i_landmark-1)==i_gt:
                cv2.line(canvas,init_pt_in_canvas,init_pt_in_canvas,(0,255,0),4)
            else:
                cv2.line(canvas,init_pt_in_canvas,init_pt_in_canvas,(0,0,255),4)
    return canvas

def vis_localizations(canvas, img_t,i_landmark,step_t,all_action_list,initMark_X,gt):
    meanImg = para.meanImg
    img_t = img_t + meanImg
    img = np.asarray(img_t, np.uint8)
    
    initMark_Y = [11.0, 31.0, 51.0, 71.0, 91.0]
    cur_point = initMark_X[i_landmark-1]
    # draw init
    i_step = 0
    
    canvas = draw_single_action(canvas, i_landmark, i_step, initMark_X, initMark_Y, gt, img)
    
    # draw actions
    for i_step in range(1,step_t+1):        
        #footnote_start = (1000 * i_step, 550)        
        if all_action_list[i_step-1]==1:
            cur_point = -20
        elif all_action_list[i_step-1]==2:
            cur_point = cur_point - 5
        elif all_action_list[i_step-1]==3:
            cur_point = cur_point + 5
        elif all_action_list[i_step-1]==4:
            cur_point = cur_point
            break;
            #initMark_X = updateInitMarkByPriors(initMark_X, k, lane_class_t)

        initMark_X[i_landmark-1] = cur_point
        # print initMark_X
        # pdb.set_trace()
        text_start = (margin*(i_step-1)+50+75,margin*(num_of_landmarks-i_landmark)+50+120)
        actStr = 'Step'+ str(i_step) + ': ' + ActionStr[int(all_action_list[i_step-1])]
        canvas = cv2.putText(canvas,actStr, text_start, cv2.FONT_ITALIC, 0.5, (0,0,0),thickness=1, lineType=8)
        canvas = draw_single_action(canvas, i_landmark, i_step, initMark_X, initMark_Y, gt, img)
    return canvas


def updateInitMarkByPriors(initMark_X_t, k, lane_class_t):

    return initMark_X_t

def testAgent(model, lane_class_t, initMark_X_t, k, landmark_fea_trans_t, gt, img_t, canvas):
    cur_point_t = initMark_X_t[k-1]
    if para.num_of_history == 0:
        hist_vec_t = []
    else:
        hist_vec_t = np.zeros([len_of_history_vector])
    state_t = get_state(cur_point_t,hist_vec_t)
    all_action_list = np.zeros(num_of_steps)
    status_t = 1
    action_t = []
    step_t = 0
    qval_t = 0
    while (status_t == 1) & (step_t < num_of_steps):
        step_t += 1
        qval_t = model.predict([landmark_fea_trans_t,state_t.T], batch_size=1)
        action_t = (np.argmax(qval_t))+1
        all_action_list[step_t-1] = action_t
        # movement action, make the proper zoom on the image
        if action_t != 4:
            if action_t == 1:
                cur_point_t = -20
            elif action_t == 2:
                cur_point_t = cur_point_t - 5
            elif action_t == 3:
                cur_point_t = cur_point_t + 5

        if action_t == 4:
            status_t = 0
        if para.num_of_history != 0:
            hist_vec_t = update_history_vector(hist_vec_t, action_t)
        state_t = get_state(cur_point_t,hist_vec_t)
    dst = abs(cur_point_t-gt[k-1])
    # updatedMark_X_t = updateInitMarkByPriors(initMark_X_t, k, lane_class_t)
    # draw
    canvas = vis_localizations(canvas, img_t,k,step_t,all_action_list,initMark_X_t,gt)
    return step_t,cur_point_t, dst, all_action_list, canvas

def testLane(matName, model, fea_t, lane_class_t, gt, img_t):

    initMark_X_t = []
    if lane_class_t == '1':
        initMark_X_t = [91.0, 71.0, 51.0, 31.0, 11.0]
    else:
        initMark_X_t = [11.0, 31.0, 51.0, 71.0, 91.0]
    canvas = np.ones((750,1500,3),dtype="uint8")*255

    det_dist = abs(initMark_X_t-gt)

    lane_steps = []
    lane_point = []
    lane_dist = []
    lane_ac_list = []

    
    for i_gt in range(0,num_of_landmarks):
        if gt[i_gt] == -1:
            gt[i_gt] = -20

    for k in np.arange(num_of_landmarks,0,-1):#[5,4,3,2,1]

        gt_point_t = gt[k-1]
        if seleted_model==2:               
            landmark_fea_t = fea_t[(k-1)*landmark_region:k*landmark_region,:,:]
        elif seleted_model==3:
            landmark_fea_t = img_t[(k-1)*20:k*20,:,:]
        landmark_fea_np_t = np.array(landmark_fea_t)
        if seleted_model==2:
            landmark_fea_trans_t = np.reshape(landmark_fea_np_t,(1,4,20,512))
        elif seleted_model==3:
            landmark_fea_trans_t = np.reshape(landmark_fea_np_t,(1,20,100,3))
        all_action_list = np.zeros(num_of_steps)
               
        step_t, cur_point_t, dst, all_action_list, canvas = \
                    testAgent(model, lane_class_t, initMark_X_t, k, landmark_fea_trans_t, gt,img_t,canvas)

        

        lane_steps.append(step_t)
        lane_point.append(cur_point_t)
        lane_dist.append(dst)
        lane_ac_list.append(all_action_list)

        # randomly show selected action lists
        if random.random() < 0.005:
            hitstr = '====Class:' + lane_class_t + ':RL:{:d},Detect:{:d},GT:{:d}'\
                                .format(int(cur_point_t),int(initMark_X_t[k-1]), int(gt_point_t))
            #output.writelines(hitstr)
            #output.writelines('\n')
            all_action_list_str = [str(i_ac) for i_ac in all_action_list]
            actionStr = '-'.join(all_action_list_str)
            #output.writelines(actionStr)
            #output.writelines('\n')
            print hitstr
            print all_action_list

    imgPath = outputVisPath + '/' + matName.split('.mat')[0] + '.png'
    cv2.imwrite(imgPath, canvas)
    return lane_steps, lane_point, lane_dist, lane_ac_list, det_dist


def eval0504(model, output, testingData, testingSetList, model_info):
    if output!=0:
        output.writelines(model_info)
        output.writelines('\n')


    dst_threshold = para.dst_threshold

    landmark_region = para.landmark_region   
    meanImg = para.meanImg

    clses_test, feas_test, imgs_test, gtes_test = testingData

    hit_cnt = 0
    detect_hit_cnt = 0
    test_cnt = 0
    sup_cnt = 0
    num_of_all_agent = len(clses_test)*num_of_landmarks

    steps_cnt = 0

    for i_test in range(len(clses_test)):
        # read data
        fea_t = feas_test[i_test]
        lane_class_t = clses_test[i_test]
        groundtruth_t = gtes_test[i_test]
        img_t = imgs_test[i_test]
        img_t = misc.imresize(img_t,[100,100,3])
        img_t = img_t - meanImg        

        # the returned values are reverse
        lane_steps, lane_point, lane_dist, lane_ac_list, det_dist = \
                    testLane(testingSetList[i_test].split('/')[-1], model, fea_t, lane_class_t, groundtruth_t, img_t)

        # cnt
        steps_cnt += np.sum(lane_steps,0)
        hit_cnt += len( np.where(np.asarray(lane_dist,np.float)<dst_threshold)[0] )
        detect_hit_cnt += len(np.where(np.asarray(det_dist,np.float) < dst_threshold)[0])
        sup_cnt += len(np.where(np.asarray((lane_dist - det_dist), np.float) <= 0)[0])
    # metrics
    avg_step = float(steps_cnt)/num_of_all_agent
    acc_hit = float(hit_cnt)/num_of_all_agent
    acc_detect = float(detect_hit_cnt)/num_of_all_agent
    acc_sup = float(sup_cnt)/num_of_all_agent
    
    reStr = 'Test RL:{:.4f}\nTest Det:{:.4f}\nRL is superior to Det:{:.2f}\nAvg steps of Agents:{:.3f}\n'\
    			.format(acc_hit, acc_detect, acc_sup, avg_step)
    if output!=0:
        output.writelines(reStr)
        output.writelines('\n\n\n\n')
    # pdb.set_trace()
    return acc_hit, acc_detect, acc_sup, avg_step, reStr



def eval(model, output, testingData, para, model_info):
    
    output.writelines(model_info)
    output.writelines('\n')

    num_of_landmarks = para.num_of_landmarks
    dst_threshold = para.dst_threshold
    num_of_steps = para.num_of_steps
    len_of_history_vector = para.num_of_actions*para.num_of_history

    landmark_region = para.landmark_region

    seleted_model = para.seleted_model
    meanImg = para.meanImg

    clses_test, feas_test, imgs_test, gtes_test = testingData

    hit_cnt = 0
    detect_hit_cnt = 0
    test_cnt = 0
    sup_cnt = 0

    steps_cnt = 0

    for i_test in range(len(clses_test)):
        # read data
        fea_t = feas_test[i_test]
        # print np.sum(fea_t,axis=2)

        lane_class_t = clses_test[i_test]
        groundtruth_t = gtes_test[i_test]
        img_t = imgs_test[i_test]
        img_t = misc.imresize(img_t,[100,100,3])
        img_t = img_t - meanImg
        initMark_X_t = []# init values on X aixs
        if lane_class_t == '1':
            initMark_X_t = [91.0, 71.0, 51.0, 31.0, 11.0]
        else:
            initMark_X_t = [11.0, 31.0, 51.0, 71.0, 91.0]

        # five landmarks 
        for k in np.arange(num_of_landmarks,0,-1):#[5,4,3,2,1]
            #
            #if gt_point_t == -1:
            #    continue;
            if groundtruth_t[k-1] == -1:
                groundtruth_t[k-1] = -20
            gt_point_t = groundtruth_t[k-1]

            absolute_status_t = 1
            action_t = []
            step_t = 0
            qval_t = 0

            if seleted_model==2:               
                landmark_fea_t = fea_t[(k-1)*landmark_region:k*landmark_region,:,:]
            elif seleted_model==3:
                landmark_fea_t = img_t[(k-1)*20:k*20,:,:]
            landmark_fea_np_t = np.array(landmark_fea_t)
            if seleted_model==2:
                landmark_fea_trans_t = np.reshape(landmark_fea_np_t,(1,4,20,512))
            elif seleted_model==3:
                landmark_fea_trans_t = np.reshape(landmark_fea_np_t,(1,20,100,3))
            all_action_list = np.zeros(num_of_steps)

            status_t = 1
            cur_point_t = initMark_X_t[k-1]
            if para.num_of_history==0:
                hist_vec_t = []
            else:
                hist_vec_t = np.zeros([len_of_history_vector])
            state_t = get_state(cur_point_t,hist_vec_t)

            while (status_t == 1) & (step_t < num_of_steps):
                step_t += 1
                qval_t = model.predict([landmark_fea_trans_t,state_t.T], batch_size=1)
                action_t = (np.argmax(qval_t))+1
                all_action_list[step_t-1] = action_t
                # movement action, make the proper zoom on the image
                if action_t != 4:
                    if action_t == 1:
                        cur_point_t = -20
                    elif action_t == 2:
                        cur_point_t = cur_point_t - 5
                    elif action_t == 3:
                        cur_point_t = cur_point_t + 5

                if action_t == 4:
                    status_t = 0
                    #if step_t == 1:
                    #    absolute_status_t = 0
                if para.num_of_history!=0:
                    hist_vec_t = update_history_vector(hist_vec_t, action_t)
                state_t = get_state(cur_point_t,hist_vec_t)
            steps_cnt = steps_cnt + step_t
            # final point
            final_point = cur_point_t
            final_dist = abs(final_point-gt_point_t)          

            if random.random() < 0.005:
                hitstr = '====Class:' + lane_class_t + ':RL:{:d},Detect:{:d},GT:{:d}'.format(int(final_point),int(initMark_X_t[k-1]), int(gt_point_t))
                output.writelines(hitstr)
                output.writelines('\n')
                #pdb.set_trace()
                all_action_list_str = [str(i_ac) for i_ac in all_action_list]
                actionStr = '-'.join(all_action_list_str)
                output.writelines(actionStr)
                output.writelines('\n')
                print hitstr
                print all_action_list
            #print all_action_list
            #pdb.set_trace()
            if final_dist < dst_threshold:
                hit_cnt += 1

            # detect Results
            det_dst = abs(initMark_X_t[k-1]-gt_point_t)
            if det_dst < dst_threshold:
                detect_hit_cnt += 1
            test_cnt += 1

            if final_dist<=det_dst:
                sup_cnt +=1

            #pdb.set_trace()
    reStr = 'Test RL:{:.4f}\nTest Det Results:{:.4f}\nRL is superior to Det:{:.2f}\n'\
                .format(float(hit_cnt)/test_cnt,\
                    float(detect_hit_cnt)/test_cnt, \
                    float(sup_cnt)/test_cnt)
    avg_step_of_agent = float(steps_cnt)/len(clses_test)/5
    stepStr = 'Avg steps of Agents:{:.3f}\n'.format(avg_step_of_agent)
    output.writelines(reStr)
    output.writelines(stepStr)
    output.writelines('\n\n\n\n')
    return hit_cnt, detect_hit_cnt, sup_cnt, test_cnt, avg_step_of_agent


    
    
