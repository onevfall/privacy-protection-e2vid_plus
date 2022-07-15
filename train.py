from re import I
import torch
from utils.loading_utils import load_model, get_device
import numpy as np
import argparse
import pandas as pd
from utils.event_readers import FixedSizeEventReader, FixedDurationEventReader
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from utils.timers import CudaTimer
import time 
from image_reconstructor import ImageReconstructor
from options.inference_options import set_inference_options
from model.model import *
#新进的库
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from torch.autograd import Variable
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def print_model_grad(epoch, model, str):
    grad_file = open(join(args.output_folder, 'train_grad.txt'), 'a')
    grad_file.write('{}:\n'.format(str))
    grad_file.write('epoch:{}\n'.format(epoch))
    for name, weight in model.named_parameters():
        if weight.requires_grad: #True
            # print("weight grad:", weight.grad) # 打印梯度，看是否丢失
            # # 直接打印梯度会出现太多输出，可以选择打印梯度的均值、极值，但如果梯度为None会报错
            # print("weight.grad:", weight.grad.mean(), weight.grad.min(), weight.grad.max())
            grad_file.write('weight grad:{}\n'.format(weight.grad))
            #grad_file.write('weight grad:{} {} {}\n'.format(weight.grad.mean(), weight.grad.min(), weight.grad.max() )) 
    #file.close()
def print_model_weight(epoch, model, str):
    weight_file = open(join(args.output_folder, 'train_weight.txt'), 'a')
    weight_file.write('{}:\n'.format(str))
    weight_file.write('epoch:{}\n'.format(epoch))
    count = 0
    for name, weight in model.named_parameters():
        count += 1
        if count <= 1:  #只打印一个
            weight_file.write('weight:{}\n'.format(weight))
 # 1. 根据网络层的不同定义不同的初始化方式     
def weight_init(m):
    if isinstance(m, nn.Linear):
        #print(111)
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式 
    elif isinstance(m, nn.Conv2d):
        #print(222)
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        #print(333)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
def postprocess(data):
    #三列，第一列为0-240，第二列为0-180，第三列为0或1
    #xs
    xs = data[:, 0]
    max_xs = torch.max(xs)
    min_xs = torch.min(xs)
    data[:, 0] = (xs - min_xs) / (max_xs - min_xs) * 240
    #ys
    ys = data[:, 1]
    max_ys = torch.max(ys)
    min_ys = torch.min(ys)
    data[:, 1] = (ys - min_ys) / (max_ys - min_ys) * 180    
    
    #pols维度,回归到0、1
    pols = data[:, 2]
    mean_pols = torch.mean(pols)
    pols = pols > mean_pols
    data[:, 2] = pols
    output = data
    return output
class fd_Net(nn.Module):  #对于处理这部分[5,180,240],可加入一维
    #对于多通道数据 h * w * channel,用卷积核 1 * 1 * channel * filter 进行卷积
    def __init__(self):
        super(fd_Net,self).__init__()
        #1.暂不使用卷积层 2.使用两个fc层

        self.fc1 = nn.Linear(5*180*240,64)
        self.fc2 = nn.Linear(64,5*180*240)
        self.relu = nn.ReLU(inplace=False)
        #self.dropout = nn.Dropout(0.5) 加dropout会报错
    def forward(self,input):
        x = torch.flatten(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        output = x.reshape(5,180,240)
        return output
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluating a trained network')
    
    parser.add_argument('-c', '--path_to_model', required=True, type=str,
                       help='path to model weights')
    parser.add_argument('-i', '--input_file', required=True, type=str)
    parser.add_argument('--fixed_duration', dest='fixed_duration', action='store_true')
    parser.set_defaults(fixed_duration=False)
    parser.add_argument('-N', '--window_size', default=None, type=int,
                        help="Size of each event window, in number of events. Ignored if --fixed_duration=True")
    parser.add_argument('-T', '--window_duration', default=33.33, type=float,
                        help="Duration of each event window, in milliseconds. Ignored if --fixed_duration=False")
    parser.add_argument('--num_events_per_pixel', default=0.35, type=float,
                        help='in case N (window size) is not specified, it will be \
                              automatically computed as N = width * height * num_events_per_pixel')
    parser.add_argument('--skipevents', default=0, type=int)
    parser.add_argument('--suboffset', default=0, type=int)
    parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
    parser.set_defaults(compute_voxel_grid_on_cpu=False)
    '''新加的'''
    parser.add_argument('--train_model', dest='train_model', action='store_true') #增加对于函数
    parser.set_defaults(train_model=False)
    #若fd_train为true，则固定E2VID，训练退化fd_net；若为false，则固定fd_net,训练E2VID
    parser.add_argument('--fd_train', dest='fd_train', action='store_true') 
    parser.set_defaults(fd_train=False)  
    parser.add_argument("--epoches", default=20, type=int, help="The number of epochs")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="learning_rate")

    set_inference_options(parser)

    args = parser.parse_args()

    # Read sensor size from the first first line of the event file
    path_to_events = args.input_file

    header = pd.read_csv(path_to_events, delim_whitespace=True, header=None, names=['width', 'height'],
                         dtype={'width': int, 'height': int},
                         nrows=1)
    width, height = header.values[0]
    print('Sensor size: {} x {}'.format(width, height))  #(240,180)
    #本代码，运行两个并行的model框架进行，一个以另一个的结果进行train

    # Load model
    device = get_device(args.use_gpu)
    model_image = load_model(args.path_to_model, device)
    model_image = model_image.to(device)
    model_image.eval()  #调为eval模式,但使用backward仍然会更新参数
    #冻结模型
    for layer in list(model_image.parameters()):
        layer.requires_grad=False  #设置之后打印参数是没有显示requires_grad参数的，说明参数梯度冻结
        
    # #原模型的相关参数，可以这样直接初始化，但是太丑了
    # config={'num_bins': 5, 
    #         'skip_type': 'sum', 
    #         'recurrent_block_type': 'convlstm', 
    #         'num_encoders': 3, 
    #         'base_num_channels': 32, 
    #         'num_residual_blocks': 2, 
    #         'use_upsample_conv': False, 
    #         'norm': 'BN'}
    # # instantiate model
    # model_train = eval('E2VIDRecurrent')(config)

    model_train = load_model(args.path_to_model, device) #结构和权重先加载与image一样的
    fd_net = fd_Net().to(device)  #退化模型放上gpu
    
    if False == args.train_model: #加载训练的模型权重，进行测试
        #model = torch.load('train/trained_model/model.pth')
        model_train.load_state_dict(torch.load("train/trained_E2VID_model/2022-06-01 19:49:10.412563+08:00model_param.pkl")) 
        print('已加载训练模型权重')
        print('进入测试模式')
        model_train.eval()
    elif False == args.fd_train: 
        #若fd非train模式，则此时是训练E2VID模型，那么随机初始化；
        #若fd为train模式，则E2VID直接加载与image一样的，此处不改变即可，默认就是加载与image一样的
        model_train.apply(weight_init)  #随机初始化,打乱权重
        model_train.train()
        print('进入训练模式:训练E2VID')
        #加载fd_net的权重
        fd_net.load_state_dict(torch.load("train/trained_fd_model/2022-06-01 16:40:43.909909+08:00model_param.pkl")) 
        
        #将fd_net置为不训练
        for layer in list(fd_net.parameters()):
            layer.requires_grad=False
        print_model_weight(-1, fd_net,'fd_net')
    else:
        print('进入训练模式:训练fd_net')
        print_model_weight(-1, model_train,'model_train')
    print_model_weight(-1, model_image,'model_image')
    model_train = model_train.to(device) 
    
    """ Read chunks of events using Pandas """

    # Loop through the events and reconstruct images
    N = args.window_size  #一个window的大小
    if not args.fixed_duration:
        if N is None:
            #N在这儿进行了window_size的初始化，指出了一个window的装载的事件量，由W,H,每个像素点的事件量定义
            N = int(width * height * args.num_events_per_pixel)
            print('Will use {} events per tensor (automatically estimated with num_events_per_pixel={:0.2f}).'.format(
                N, args.num_events_per_pixel))
        else:
            print('Will use {} events per tensor (user-specified)'.format(N))
            mean_num_events_per_pixel = float(N) / float(width * height)
            if mean_num_events_per_pixel < 0.1:
                print('!!Warning!! the number of events used ({}) seems to be low compared to the sensor size. \
                    The reconstruction results might be suboptimal.'.format(N))
            elif mean_num_events_per_pixel > 1.5:
                print('!!Warning!! the number of events used ({}) seems to be high compared to the sensor size. \
                    The reconstruction results might be suboptimal.'.format(N))
    
    initial_offset = args.skipevents
    sub_offset = args.suboffset
    start_index = initial_offset + sub_offset  #此处default，为0
    
    if args.compute_voxel_grid_on_cpu:
        print('Will compute voxel grid on CPU.')

    if args.fixed_duration:
        event_window_iterator = FixedDurationEventReader(path_to_events,
                                                         duration_ms=args.window_duration,
                                                         start_index=start_index)
    else:
        #执行此句，从数据集中读取事件流数据并处理为窗口格式！我们在这之前进行一个退化变换？那就可能会断掉训练、影响原数据
        #所以我们在处理为窗口格式后再进行变换最好
        event_window_iterator = FixedSizeEventReader(path_to_events,options = args , num_events=N, start_index=start_index)
    
    #定义优化器,将两个参数都融入
    optimizer = torch.optim.Adam([{"params":fd_net.parameters()},
    {"params":model_train.parameters()}], lr=args.learning_rate)
    if args.fd_train: #只训练fd，则可以去除model_train进行实验
        optimizer = torch.optim.Adam([{"params":fd_net.parameters()}], lr=args.learning_rate)
    else:  #固定fd，只训练model_train
        optimizer = torch.optim.Adam([{"params":model_train.parameters()}], lr=args.learning_rate)
    
    count = 0
    #     训练模式的时间
    if args.train_model:  #训练模式
        #取消计时
        for epoch in tqdm(range(0, args.epoches)):
            count = 0 #每次初始化一次
            
            #每次epoch后迭代器和索引都要更新，才能进行下一轮重复
            start_index = initial_offset + sub_offset
            #此处增加了一个options，方便传参不要输出
            event_window_iterator = FixedSizeEventReader(path_to_events, options = args , num_events=N, start_index=start_index)
            #遍历每个窗口，对于每个窗口，event_window.shape=(15119, 4),4是指t,x,y,p的四个维度
            for event_window in event_window_iterator: 
                                     
                #num_bins=5
                #每个窗口都要更新这个reconstructor的地方，但调整到这儿，原本输出的很多内容会有点乱
                reconstructor_train = ImageReconstructor(model_train, height, width, model_train.num_bins, args)
                reconstructor_image = ImageReconstructor(model_image, height, width, model_image.num_bins, args)
                last_timestamp = event_window[-1, 0]
                #此处默认在GPU上跑的，划分体素网格,内部已禁用梯度计算
                event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                            num_bins=model_train.num_bins,
                                                            width=width,
                                                            height=height,
                                                            device=device)
                num_events_in_window = event_window.shape[0] #15119
                image = reconstructor_image.update_reconstruction(event_tensor, start_index + num_events_in_window, last_timestamp)
                image = image.type(torch.float).to(device)

                train_event_tensor = event_tensor.type(torch.float)
                train_event_tensor = train_event_tensor.to(device)
                train_event_tensor.requires_grad = True
                # torch.set_printoptions(profile="full")
                # print(train_event_tensor.shape) #torch.Size([5, 180, 240])
                #退化模型处理
                train_event_tensor = fd_net(train_event_tensor)   
                
                output = reconstructor_train.update_reconstruction(train_event_tensor, start_index + num_events_in_window, last_timestamp)
                output = output.type(torch.float).to(device)
                
                #image.shape: torch.Size([180, 240])
                #output.shape: torch.Size([180, 240])
                loss_func = nn.MSELoss() #psnr = 10 * np.log10(255 * 255 / mse) #psnr越小越差，不能直接作为评估标准
                if args.fd_train: 
                    loss = -loss_func(output,image) #取负
                else:
                    loss = loss_func(output,image)
                #反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                start_index += num_events_in_window  #下一个窗口
            # #打印grad，训练前和训练时
            # print_model_weight(epoch, model_train,'model_train')
            # print_model_weight(epoch, model_image,'model_image')
            # #print_model(epoch, model_train)

            print_model_weight(epoch, model_image,'model_image')
            if args.fd_train: 
                print(epoch,'fd_train loss:',loss.item())
                loss_file = open(join(args.output_folder, 'fd_loss.txt'), 'a')
                loss_file.write('epoch:{}  loss:{:.8f}\n'.format(epoch, loss))
                #此时看E2VID网络的情况
                print_model_weight(epoch, model_train,'model_train')
            else:
                print(epoch,'E2VID_train loss:',loss.item())
                loss_file = open(join(args.output_folder, 'E2VID_loss.txt'), 'a')
                loss_file.write('epoch:{}  loss:{:.8f}\n'.format(epoch, loss))
                #此时看fd_net网络的情况
                print_model_weight(epoch, fd_net,'fd_net')
    else:
        reconstructor = ImageReconstructor(model_train, height, width, model_train.num_bins, args)
        with CudaTimer('Processing entire dataset'):  #有计时作用
            #遍历每个窗口，对于每个窗口，event_window.shape=(15119, 4),4是指t,x,y,p的四个维度
            for event_window in event_window_iterator: 
                last_timestamp = event_window[-1, 0]
                
                with CudaTimer('Building event tensor'):
                    if args.compute_voxel_grid_on_cpu:
                        event_tensor = events_to_voxel_grid(event_window,
                                                            num_bins=model_train.num_bins,
                                                            width=width,
                                                            height=height)
                        event_tensor = torch.from_numpy(event_tensor)
                    else: #在GPU上跑的，划分体素网格
                        event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                                    num_bins=model_train.num_bins,
                                                                    width=width,
                                                                    height=height,
                                                                    device=device)
                    '''这里可以用论文辅助理解events_to_voxel_grid_pytorch，对于num_bins也在论文里有解释！！！'''
                num_events_in_window = event_window.shape[0] #15119
                reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window, last_timestamp)
                                    
                start_index += num_events_in_window  #下一个窗口
    # train模式下，保存整个模型
    if args.train_model:  
        #记录时间，分开记录训练记录
        SHA_TZ = timezone(
        timedelta(hours=8),
        name='Asia/Shanghai',
        )
        # 协调世界时
        utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
        # 当前北京时间
        beijing_now = utc_now.astimezone(SHA_TZ)
        ts = str(beijing_now)
        if args.fd_train:  #此时训练fd
            torch.save(fd_net, 'train/trained_fd_model/'+ts+'model.pth')   
            torch.save(fd_net.state_dict(), 'train/trained_fd_model/'+ts+'model_param.pkl')
            print("fd模型训练完成，已保存模型内容")
        else:     #此时训练E2VID
            torch.save(model_train, 'train/trained_E2VID_model/'+ts+'model.pth')   
            torch.save(model_train.state_dict(), 'train/trained_E2VID_model/'+ts+'model_param.pkl')
            print("E2VID模型训练完成，已保存模型内容")
    else:
        print("测试的重建完成")