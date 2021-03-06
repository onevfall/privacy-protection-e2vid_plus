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
torch.set_printoptions(profile="full")  #完整打印

def hook_fn(grad):
    grad_file = open(join(args.output_folder, 'nonleaf_grad.txt'), 'a')
    grad_file.write('grad:{}\n'.format(grad))
def print_model_grad(epoch, model, str):
    grad_file = open(join(args.output_folder, 'train_grad.txt'), 'a')
    grad_file.write('{}:\n'.format(str))
    grad_file.write('epoch:{}\n'.format(epoch))
    count = 0
    for name, param in model.named_parameters():
        #while(count<2): #只打印两轮
        if param.requires_grad: #True
            # # 直接打印梯度会出现太多输出，可以选择打印梯度的均值、极值，但如果梯度为None会报错
            # print("param.grad:", param.grad.mean(), param.grad.min(), param.grad.max())
            count += 1
            grad_file.write('param{} grad:{}\n'.format(name,param.grad))
            # grad_file.write('param grad mean:{} min:{} max:{}\n'.format(param.grad.mean(), param.grad.min(), param.grad.max() )) 
    #file.close()
def print_model_weight(epoch, model, str):
    weight_file = open(join(args.output_folder, 'train_weight.txt'), 'a')
    weight_file.write('{}:\n'.format(str))
    weight_file.write('epoch:{}\n'.format(epoch))
    count = 0
    for name, weight in model.named_parameters():
        count += 1
        if count <= 1: #只输出一次，减少输出量
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
def postprocess(data): #数据恢复原有格式
    #三列，第一列为0-240，第二列为0-179，第三列为0或1
    #xs
    xs = data[:, 0]
    max_xs = xs.max().clone().detach() #这样使用解决
    min_xs = xs.min().clone().detach() 
    data[:, 0] = (xs - min_xs) / (max_xs - min_xs) * 239  #最大值不是240，是239.......
    data[:, 0] = torch.floor(data[:, 0])  #向下取整
    #ys
    ys = data[:, 1]
    max_ys = ys.max().clone().detach()  #torch.max(ys)
    min_ys = ys.min().clone().detach()  #torch.min(ys) 
    data[:, 1] = (ys - min_ys) / (max_ys - min_ys) * 179    
    data[:, 1] = torch.floor(data[:, 1])
    #pols维度,回归到0、1
    pols = data[:, 2]
    mean_pols = torch.mean(pols)
    
    #pols = [1 if x > mean_pols else 0 for x in pols ] 列表推导式，是python的处理,不可用
    pols = pols > mean_pols  #输出均为True，False这种,看结果确实是回到了0、1

    data[:, 2] = pols
    output = data
    return output
class fd_Net(nn.Module):
    def __init__(self):
        super(fd_Net,self).__init__()
        #1.是否会使用卷积层 2.fc层如何定义维度数量 3.此处只对于x,y,p做一个训练
        self.fc1 = nn.Linear(3,64)
        self.fc2 = nn.Linear(64,3)
        self.relu = nn.ReLU(inplace=False)
        #self.dropout = nn.Dropout(0.5) 加dropout会报错
    def forward(self,input):
        #print('input.requires_grad:',input.requires_grad)
        input_1 = input[:,:1] #此列数据不变
        #input_1.register_hook(hook_fn)  #这部分居然input_1有grad，本不应该有？
        x0 = input[:,1:]
        x = self.fc1(x0)
        #x.register_hook(hook_fn)
        x1 = self.relu(x) #使用clone防止占位，之前是使用ReLU
        #x = self.dropout(x)
        x1 = self.fc2(x1)
        x1 = postprocess(x1)
        
        #x1.register_hook(hook_fn)  #grad.shape:torch.Size([15119, 3])
        output = torch.cat([input_1,x1],dim=1)
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
    parser.add_argument("--epoches", default=2, type=int, help="The number of epochs")
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
    device = get_device(args.use_gpu)  #此处打印device1的归属
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

    #直接用model_path加载也试试

    model_train = load_model(args.path_to_model, device) #结构和权重先加载之前的
    if False == args.train_model: #加载训练的模型权重，进行测试
        #model = torch.load('train/trained_model/model.pth')
        model_train.load_state_dict(torch.load("train/trained_E2VID_model/2022-05-19 01:34:28.303251+08:00model_param.pkl")) 
        print('已加载训练模型权重，开始测试')
        model_train.eval()
    elif False == args.fd_train: #若fd非train模式，则此时是训练E2VID模型，那么随机初始化；若fd为train模式，则E2VID直接加载与image一样的
        model_train.apply(weight_init)  #随机初始化,打乱权重
        model_train.train()
        print("对E2VID模型进入训练模式")
    else:
        print("对fd_net进入训练模式")
    model_train = model_train.to(device) 
    fd_net = fd_Net().to(device)  #退化模型放上
    fd_net.apply(weight_init)  #加载权重
    fd_net.train()  #开启训练模式
    #fd_net = fd_net.double() #与数据统一，转化为double进行处理
    if False == args.fd_train:
        for layer in list(fd_net.parameters()):
            layer.requires_grad=False

    #打印grad，训练前和训练时
    # print_model_weight(-1, model_train,'model_train')
    # print_model_weight(-1, fd_net,'fd_net')
    
    # print_model_grad(-1,model_train,'model_train')
    # print_model_grad(-1, fd_net,'fd_net')
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
        #执行此句，从数据集中读取事件流数据！我们在这之前进行一个退化变换？那就可能会断掉训练、影响原数据
        event_window_iterator = FixedSizeEventReader(path_to_events,options = args , num_events=N, start_index=start_index)
    
    #定义优化器,将两个参数都融入
    optimizer = torch.optim.Adam([{"params":fd_net.parameters()},{"params":model_train.parameters()}], lr=args.learning_rate)
    if args.fd_train: #只训练fd，则可以去除model_train进行实验
        optimizer = torch.optim.Adam([{"params":fd_net.parameters()}], lr=args.learning_rate)
    else:  #固定fd，只训练model_train
        optimizer = torch.optim.Adam([{"params":model_train.parameters()}], lr=args.learning_rate)
    
    count = 0
    #     训练模式的时间
    if args.train_model:  #训练模式
        with torch.autograd.set_detect_anomaly(True):    
            #取消计时
            for epoch in tqdm(range(0, args.epoches)):
                count = 0 #每次初始化一次
                
                #每次epoch后迭代器和索引都要更新，才能进行下一轮重复
                start_index = initial_offset + sub_offset
                #此处增加了一个options，方便传参不要输出
                event_window_iterator = FixedSizeEventReader(path_to_events, options = args , num_events=N, start_index=start_index)
                #遍历每个窗口，对于每个窗口，event_window.shape=(15119, 4),4是指t,x,y,p的四个维度
                for event_window in event_window_iterator:  
                    old_train_event_window = torch.from_numpy(event_window)
                    old_train_event_window = old_train_event_window.type(torch.float)
                    old_train_event_window = old_train_event_window.to(device)
                    #old_train_event_window.requires_grad = True
                    #退化模型处理
                    train_event_window = fd_net(old_train_event_window) 
                    # print('old_train_event_window.requires_grad:',old_train_event_window.requires_grad)                   
                    # print('train_event_window.requires_grad:',train_event_window.requires_grad)
                    #train_event_window.register_hook(hook_fn)  
                    #print('train_event_window:',train_event_window.requires_grad)                     
                    #num_bins=5
                    #每个窗口都要更新这个reconstructor的地方，但调整到这儿，原本输出的很多内容会有点乱
                    reconstructor_train = ImageReconstructor(model_train, height, width, model_train.num_bins, args)
                    reconstructor_image = ImageReconstructor(model_image, height, width, model_image.num_bins, args)
                    last_timestamp = event_window[-1, 0]
                    
                    image_event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                                num_bins=model_train.num_bins,
                                                                width=width,
                                                                height=height,
                                                                device=device)
                    #此处默认在GPU上跑的，划分体素网格,对于train模式，内部不禁用梯度计算
                    train_event_tensor = events_to_voxel_grid_pytorch(train_event_window,
                                                                num_bins=model_train.num_bins,
                                                                width=width,
                                                                height=height,
                                                                device=device)
                    #print('train_event_tensor:',train_event_tensor.requires_grad) #注意到此处打印出的tensor没有跟grad
                    
                    num_events_in_window = event_window.shape[0] #15119
                    
                    image = reconstructor_image.update_reconstruction(image_event_tensor, start_index + num_events_in_window, last_timestamp)
                    
                    output = reconstructor_train.update_reconstruction(train_event_tensor, start_index + num_events_in_window, last_timestamp)

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

                    #print('output grad:',output.grad)
                    # print(train_event_window.is_leaf)
                    # print('train_event_window grad:',train_event_window.grad.shape)

                    optimizer.step()
                    
                    start_index += num_events_in_window  #下一个窗口
                # #打印weight，训练前和训练时
                # print_model_weight(epoch, model_train,'model_train')
                # print_model_weight(epoch, fd_net,'fd_net')
                
                # print_model_grad(epoch,model_train,'model_train')
                print_model_grad(epoch, fd_net,'fd_net')
                if args.fd_train: 
                    print(epoch,'fd_train loss:',loss.item())
                    loss_file = open(join(args.output_folder, 'fd_loss.txt'), 'a')
                    loss_file.write('epoch:{}  loss:{:.8f}\n'.format(epoch, loss))
                else:
                    print(epoch,'E2VID_train loss:',loss.item())
                    loss_file = open(join(args.output_folder, 'E2VID_loss.txt'), 'a')
                    loss_file.write('epoch:{}  loss:{:.8f}\n'.format(epoch, loss))
    else: #测试模式
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
            torch.save(model_train, 'train/trained_fd_model/'+ts+'model.pth')   
            torch.save(model_train.state_dict(), 'train/trained_fd_model/'+ts+'model_param.pkl')
            print("fd模型训练完成，已保存模型内容")
        else:     #此时训练E2VID
            torch.save(model_train, 'train/trained_E2VID_model/'+ts+'model.pth')   
            torch.save(model_train.state_dict(), 'train/trained_E2VID_model/'+ts+'model_param.pkl')
            print("E2VID模型训练完成，已保存模型内容")
    else:
        print("测试的重建完成")