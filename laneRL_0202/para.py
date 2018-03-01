import os
import datetime
from scipy import io as sio
# CNN Model
modelPath = "../Lane_models"
modelName = '20170508_ly_5_dp_0_lr_0.0001_hist_1_epoch_40.h5'
seleted_model = 3#2:Featuremap model 3: image conv model

# CNN Parameters
img_batch_size_ratio = 5
batch_size = 1024
learning_rate = 1e-4
conv_layer = 5
dp=0
epochs = 100


# Data Parameters
database = '/home/anotherytongbai/Desktop/mountytongbai/ytongbai/laneDRLcode/dataset/refineFeaData_20_20_deleted/'
meanData = sio.loadmat('/mount/ytongbai/laneDRLcode/dataset/meanImg.mat')
meanImg = meanData['meanImg']
meanArray = [136.1114,132.7178,119.1245]
subDataKey = '*' + '' + '*'
dst_threshold = 5
num_of_landmarks = 5# fixed
landmark_region = 4

# Action Parameters
ActionStr = ['','Remove','Left', 'Right', 'Terminal']
num_of_actions = 4# fixed
num_of_history = 8
num_of_steps = 10

# Reward Parameters
reward_terminal_action = 3
reward_movement_action = 1
reward_invalid_movement_action = -5
reward_remove_action = 1

# RL Parameters
buffer_experience_replay = 20480*2
epochs_in_memeory = 10
gamma = 0.90
epsilon = 1

# Important Attributes
timeStr = datetime.date.today().strftime('%Y%m%d')
keyAttr = timeStr + \
			'_ly_' + str(conv_layer) + \
			'_dp_' + str(dp) + \
			'_lr_' + str(learning_rate) + \
			'_hist_' + str(num_of_history) 


logName = '../Lane_logs/' + keyAttr + '.log'
testlogName = '../Lane_logs/' + keyAttr + '_TEST.log'


#test visible
outputVisRoot = "../Lane_vis" 
if not os.path.exists(outputVisRoot):
    os.mkdir(outputVisRoot)

outputVisPath = outputVisRoot + '/' + modelName.split('.h5')[0]
if not os.path.exists(outputVisPath):
    os.mkdir(outputVisPath)


