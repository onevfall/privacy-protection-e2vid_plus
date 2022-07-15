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

    set_inference_options(parser)

    args = parser.parse_args()

    # Read sensor size from the first first line of the event file
    path_to_events = args.input_file

    header = pd.read_csv(path_to_events, delim_whitespace=True, header=None, names=['width', 'height'],
                         dtype={'width': int, 'height': int},
                         nrows=1)
    width, height = header.values[0]
    print('Sensor size: {} x {}'.format(width, height))  #(240,180)

    # Load model
    device = get_device(args.use_gpu)
    model = load_model(args.path_to_model, device)

    model = model.to(device)
    model.eval()  #调为eval模式

    #num_bins=5
    reconstructor = ImageReconstructor(model, height, width, model.num_bins, args)
    
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
        #执行此句，从数据集中读取事件流
        event_window_iterator = FixedSizeEventReader(path_to_events, num_events=N, start_index=start_index)
    
    with CudaTimer('Processing entire dataset'):  #有计时作用
        #遍历每个窗口，对于每个窗口，event_window.shape=(15119, 4),4是指t,x,y,p的四个维度
        for event_window in event_window_iterator: 
            last_timestamp = event_window[-1, 0]
            
            with CudaTimer('Building event tensor'):
                if args.compute_voxel_grid_on_cpu:
                    event_tensor = events_to_voxel_grid(event_window,
                                                        num_bins=model.num_bins,
                                                        width=width,
                                                        height=height)
                    event_tensor = torch.from_numpy(event_tensor)
                else: #在GPU上跑的，划分体素网格
                    event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                                num_bins=model.num_bins,
                                                                width=width,
                                                                height=height,
                                                                device=device)
                '''这里可以用论文辅助理解events_to_voxel_grid_pytorch，对于num_bins也在论文里有解释！！！'''
            num_events_in_window = event_window.shape[0] #15119
            
            #此时事件表示由(15119, 4)转化为event_tensor.shape = torch.Size([5, 180, 240])，
            #就是经过体素网格转化了，以方便CNN训练
            
            #程序第一次运行到这儿的时候，start_index 最初为0
            reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window, last_timestamp)
            
            start_index += num_events_in_window
