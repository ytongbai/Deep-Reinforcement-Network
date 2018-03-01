from keras.preprocessing import image
import numpy as np
from scipy import io as sio
import random
import pdb
import para
from reward import get_dst,get_reward_trigger,get_reward_movement,\
                    update_history_vector,get_reward_movement_0427,\
                    get_reward_remove

# len_of_fea_vec = para.len_of_fea_vec
num_of_actions = para.num_of_actions
num_of_history = para.num_of_history
buffer_experience_replay = para.buffer_experience_replay

def get_features(features_names):
    clses = []
    feas = []
    imgs = []
    gtes = []
    cnt = 0
    len_of_img = len(features_names)
    for i_features_files in features_names:
        cnt = cnt + 1
        if cnt%500 ==0:
            print "Loading Data: {:d}/{:d}".format(cnt,len_of_img)      
        fileData = sio.loadmat(i_features_files)
        cls = fileData['class_name'][0]
        gt = fileData['mark'][0]
        fea = fileData['fea']
        img = fileData['img']

        clses.append(cls)
        feas.append(fea)
        imgs.append(img)
        gtes.append(gt)
    print "Loading Data: Done!"
    return clses,feas,imgs,gtes
'''
def get_ori_state(fea,cur_point,hist_vec):
    fea_vec = np.reshape(fea, (len_of_fea_vec, 1))
    history_vector = np.reshape(hist_vec, (num_of_actions*num_of_history, 1))
    state = np.vstack((fea_vec, cur_point, history_vector))
    return state
'''
def get_state(cur_point,hist_vec):
    history_vector = np.reshape(hist_vec, (num_of_actions*num_of_history, 1))
    state = np.vstack((cur_point, history_vector))
    return state

def generateCNNTrainData(model,trainData):
    batch_size = para.batch_size
    gamma = para.gamma
    replay_landmark_fea_np, replay_state, replay_action, replay_reward, replay_new_state = trainData

    replay_landmark_fea_np, replay_state, 
    replay_state = np.array(replay_state)
    replay_state = replay_state.astype("float32")
    replay_landmark_fea_np = np.array(replay_landmark_fea_np)
    replay_landmark_fea_np = replay_landmark_fea_np.astype("float32")
    replay_new_state = np.array(replay_new_state)
    replay_new_state = replay_new_state.astype("float32")                

    replay_old_qval = model.predict([replay_landmark_fea_np,replay_state], batch_size=batch_size)
    replay_newQ = model.predict([replay_landmark_fea_np,replay_new_state], batch_size=batch_size)

    replay_maxQ = replay_newQ.argmax(axis=1)
    replay_y = np.zeros([batch_size,num_of_actions])
    replay_y = replay_old_qval

    mat_r_act = np.asmatrix(replay_action)
    mat_r_reward = np.asmatrix(replay_reward)
    ind_no_terminal = np.where(mat_r_act != 4)[1]# index
    flag_no_terminal = np.zeros(buffer_experience_replay)
    flag_no_terminal[ind_no_terminal] = 1
    ind_terminal = np.where(mat_r_act == 4)[1]
    flag_terminal = np.zeros(buffer_experience_replay)
    flag_terminal[ind_terminal] = 1
    # formula: act != 4: update = (reward + (gamma * maxQ))
    mat_update_no_terminal = np.multiply( flag_no_terminal, \
                                            (mat_r_reward + np.asmatrix(np.multiply(gamma,replay_maxQ))) \
                                        )
    # formula: act == 4: update = reward
    mat_update_terminal = np.multiply(flag_terminal, mat_r_reward)
    mat_update = mat_update_no_terminal + mat_update_terminal

    for i_update in range(buffer_experience_replay):
        replay_y[i_update,replay_action[i_update]-1] = mat_update[0,i_update]

    del mat_update, mat_update_terminal, mat_r_reward, mat_r_act,\
        ind_no_terminal, flag_no_terminal, ind_terminal, flag_terminal

    return replay_landmark_fea_np,replay_state,replay_y

def generateExpReplay(model, seleted_model, epsilon, replay, replay_ind, train_state, featureData, i):
    num_of_steps = para.num_of_steps
    dst_threshold = para.dst_threshold
    len_of_history_vector = para.num_of_actions*para.num_of_history

    replay_landmark_fea_np, replay_state, replay_action, replay_reward, replay_new_state = replay
    fea, initMark_X, groundtruth, img = featureData
    # five landmarks
    for k in np.arange(para.num_of_landmarks,0,-1):#[5,4,3,2,1]
        if groundtruth[k-1] == -1:
            groundtruth[k-1] = -20
        gt_point = groundtruth[k-1]

        # generate actions
        # status indicates whether the agent is still alive and has not triggered the terminal action
        status = 1
        step = 0

        cur_point = initMark_X[k-1]
        if seleted_model==2:
            landmark_fea = fea[(k-1)*para.landmark_region:k*para.landmark_region,:,:]                        
        elif seleted_model==3:            
            landmark_fea = img[(k-1)*20:k*20,:,:]

        landmark_fea_np = np.array(landmark_fea)    
        landmark_fea_trans = []

        if seleted_model==2:
            landmark_fea_trans = np.reshape(landmark_fea_np,(1,4,20,512))
        elif seleted_model==3:
            landmark_fea_trans = np.reshape(landmark_fea_np,(1,20,100,3))
        
        if para.num_of_history==0:
            hist_vec = []
        else:
            hist_vec = np.zeros([len_of_history_vector])
        state = get_state(cur_point,hist_vec)
        
        cur_dst = get_dst(gt_point,cur_point)

        last_point = cur_point
        last_dst = cur_dst
        while (status == 1) & (step < num_of_steps):

            reward = []
            qval = []
            if seleted_model==0:
                qval = model.predict(state.T, batch_size=1)
            else:
                #pdb.set_trace()
                qval = model.predict([landmark_fea_trans,state.T], batch_size=1)

            step += 1
            # we force terminal action in case actual IoU is higher than 0.5, to train faster the agent
            #print 'GTP:{:f}'.format(gt_point)
            #print 'LastP:{:f},LastD:{:f}'.format(last_point,last_dst)
            if cur_dst < dst_threshold:
                action = 4
            # epsilon-greedy policy
            elif random.random() < epsilon:
                action = np.random.randint(1, 5)
            else:
                action = (np.argmax(qval))+1
            # print action
            # terminal action
            if action == 4:
                reward = get_reward_trigger(cur_dst)
                #step += 1
                # movement action, we perform the crop of the corresponding subregion
            elif action == 1:
                cur_point = -20
                cur_dst = get_dst(gt_point,cur_point)
                reward = get_reward_remove(cur_dst)
                last_dst = cur_dst
                last_point = cur_point
            else:
                if action == 2:# to left
                    cur_point = cur_point - 5
                elif action == 3:# to right
                    cur_point = cur_point + 5
                cur_dst = get_dst(gt_point,cur_point)
                reward = get_reward_movement_0427(cur_point,last_point,gt_point)
                last_dst = cur_dst
                last_point = cur_point

            if para.num_of_history!=0:
                hist_vec = update_history_vector(hist_vec, action)
            new_state = get_state(cur_point, hist_vec)

            if len(replay_state) < buffer_experience_replay:# repaly is not stil full
                # replay.append((landmark_fea_trans, state, action, reward, new_state))
                replay_landmark_fea_np.append(landmark_fea_np)  
                replay_state.append(state[:,0])
                replay_action.append(action)
                replay_reward.append(reward)
                replay_new_state.append(new_state[:,0])
                train_state = 0
                if len(replay_state)%(buffer_experience_replay/5)==0:
                    print 'Fill RL Data in memory: {:d}% (ImgEpoch:{:d}).'\
                            .format(int(len(replay_state)*100/buffer_experience_replay),i)
            else:# repaly is full
                if replay_ind < (buffer_experience_replay-1):
                    #pdb.set_trace()
                    replay_ind += 1
                    train_state = 0
                    if (replay_ind+1)%(buffer_experience_replay/5)==0:
                        print 'Replace RL Data in memory: {:d}% (ImgEpoch:{:d}).'\
                                .format(int((replay_ind+1)*100/buffer_experience_replay),i)
                else:
                    replay_ind = -1
                    train_state = 1
                    break
                #pdb.set_trace()
                replay_landmark_fea_np[replay_ind] = landmark_fea_np
                replay_state[replay_ind] = state[:,0]
                replay_action[replay_ind] = action
                replay_reward[replay_ind] = reward
                replay_new_state[replay_ind] = new_state[:,0]     
                
            if action == 4:
                status = 0
            state = new_state
        if train_state == 1:
            break
    replay = [replay_landmark_fea_np, replay_state, replay_action, replay_reward, replay_new_state]
    return replay_ind, train_state, replay