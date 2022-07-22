import torch
import cv2
import numpy as np
from model.model import *
from utils.inference_utils import CropParameters, EventPreprocessor, IntensityRescaler, ImageFilter, ImageDisplay, ImageWriter, UnsharpMaskFilter
from utils.inference_utils import upsample_color_image, merge_channels_into_color_image  # for color reconstruction
from utils.util import robust_min, robust_max
from utils.timers import CudaTimer, cuda_timers,Timer
from os.path import join
from collections import deque
import torch.nn.functional as F


class ImageReconstructor:
    def __init__(self, model, height, width, num_bins, options): #此处的options是args

        self.model = model
        self.use_gpu = options.use_gpu
        self.device = torch.device('cuda:0') if self.use_gpu else torch.device('cpu')
        self.height = height
        self.width = width
        self.num_bins = num_bins
        self.train_model = options.train_model
        self.options = options
        self.initialize(self.height, self.width, options)

    # def hook_fn(grad):
    #     grad_file = open(join(self.options.output_folder, 'nonleaf_grad.txt'), 'a')
    #     grad_file.write('grad:{}\n'.format(grad))

    def initialize(self, height, width, options):
        if self.train_model == False:     
            print('== Image reconstruction == ')
            print('Image size: {}x{}'.format(self.height, self.width))
        self.no_recurrent = options.no_recurrent
        if self.no_recurrent:
            print('!!Recurrent connection disabled!!')

        self.perform_color_reconstruction = options.color  # whether to perform color reconstruction (only use this with the DAVIS346color)
        if self.perform_color_reconstruction:
            if options.auto_hdr:
                print('!!Warning: disabling auto HDR for color reconstruction!!')
            options.auto_hdr = False  # disable auto_hdr for color reconstruction (otherwise, each channel will be normalized independently)

        self.crop = CropParameters(self.width, self.height, self.model.num_encoders) #对图片进行裁剪的方式
        #self.model.num_encoders: 3
        #self.width:240
        #self.height:180
        
        self.last_states_for_each_channel = {'grayscale': None}

        if self.perform_color_reconstruction:
            self.crop_halfres = CropParameters(int(width / 2), int(height / 2),
                                               self.model.num_encoders)
            for channel in ['R', 'G', 'B', 'W']:
                self.last_states_for_each_channel[channel] = None
        #用args将各个类进行了实例化
        self.event_preprocessor = EventPreprocessor(options) 
        self.intensity_rescaler = IntensityRescaler(options)
        self.image_filter = ImageFilter(options)
        self.unsharp_mask_filter = UnsharpMaskFilter(options, device=self.device)
        self.image_writer = ImageWriter(options)
        self.image_display = ImageDisplay(options)

    def update_reconstruction(self, event_tensor, event_tensor_id, stamp=None):
        # with torch.no_grad(): 我们要训练，则此处不能继续这样
        if self.train_model:
            with CudaTimer('Reconstruction'):
            
                with CudaTimer('NumPy (CPU) -> Tensor (GPU)'):
                
                    #在第0维加维度
                    events = event_tensor.unsqueeze(dim=0)
                    #events.shape由torch.Size([5, 180, 240])转为torch.Size([1, 5, 180, 240])
                    events = events.to(self.device)
                # grad_file = open(join(self.options.output_folder, 'middle_events_grad.txt'), 'a')
                # grad_file.write('events grad:{}\n'.format(events.grad))
                #print('events grad:',events.grad)  #打印为None
                #预处理：去除热像素，事件张量的正常化。或翻转事件张量。
                events = self.event_preprocessor(events)  #此处调用__call__函数
                #events.shape: torch.Size([1, 5, 180, 240])

                    
                # Resize tensor to [1 x C x crop_size x crop_size] by applying zero padding
                events_for_each_channel = {'grayscale': self.crop.pad(events)} #进行裁剪和微调

                #self.crop.pad(events)：torch.Size([1, 5, 184, 240])
                reconstructions_for_each_channel = {}
                if self.perform_color_reconstruction:  #原代码版本未执行
                    events_for_each_channel['R'] = self.crop_halfres.pad(events[:, :, 0::2, 0::2])
                    events_for_each_channel['G'] = self.crop_halfres.pad(events[:, :, 0::2, 1::2])
                    events_for_each_channel['W'] = self.crop_halfres.pad(events[:, :, 1::2, 0::2])
                    events_for_each_channel['B'] = self.crop_halfres.pad(events[:, :, 1::2, 1::2])
                    
                # 此处已调用模型进行训练，建立新图
                # Reconstruct new intensity image for each channel (grayscale + RGBW if color reconstruction is enabled)
                for channel in events_for_each_channel.keys():
                    #对于auto_hdr类型的运行代码来说，只有channel：graysca
                    #print('states before trained by model:',self.last_states_for_each_channel[channel]) #答案为None
                    with CudaTimer('Inference'):
                        new_predicted_frame, states = self.model(events_for_each_channel[channel],
                                                                    self.last_states_for_each_channel[channel])
                        #new_predicted_frame torch.Size([1, 1, 184, 240])
                        #states是list,type(states)=<class 'list'>
                        #states里的元素：<generator object ImageReconstructor.update_reconstruction.<locals>.<genexpr> at 0x7fa8d32eec10>
                    if self.no_recurrent:
                        self.last_states_for_each_channel[channel] = None
                    else:
                        #运行的这个
                        self.last_states_for_each_channel[channel] = states
                        #print('states\'type after trained by model:',type(states))
                        
                    # Output reconstructed image
                    crop = self.crop if channel == 'grayscale' else self.crop_halfres

                    # Unsharp mask (on GPU)
                    new_predicted_frame = self.unsharp_mask_filter(new_predicted_frame)

                    
                    # Intensity rescaler (on GPU)
                    new_predicted_frame = self.intensity_rescaler(new_predicted_frame)

                    # print('new_predicted_frame grad:',new_predicted_frame.grad)

                    with Timer('Tensor (GPU) -> NumPy (CPU)'):
                        reconstructions_for_each_channel[channel] = new_predicted_frame[0, 0, crop.iy0:crop.iy1,
                                                                                        crop.ix0:crop.ix1]# .cpu().numpy() 去掉这，保持torch
                if self.perform_color_reconstruction:
                    out = merge_channels_into_color_image(reconstructions_for_each_channel)  #没用到，暂时没改里面的np
                else:
                    out = reconstructions_for_each_channel['grayscale']
        else:
            with torch.no_grad():     #减少cuda消耗
                with CudaTimer('Reconstruction'):
                    with CudaTimer('NumPy (CPU) -> Tensor (GPU)'):
                    
                        #在第0维加维度
                        events = event_tensor.unsqueeze(dim=0)
                        #events.shape由torch.Size([5, 180, 240])转为torch.Size([1, 5, 180, 240])
                        events = events.to(self.device)
                        
                    #预处理：去除热像素，事件张量的正常化。或翻转事件张量。
                    events = self.event_preprocessor(events)  #此处调用__call__函数
                    #events.shape: torch.Size([1, 5, 180, 240])
                    
                    # Resize tensor to [1 x C x crop_size x crop_size] by applying zero padding
                    events_for_each_channel = {'grayscale': self.crop.pad(events)} #进行裁剪和微调

                    #self.crop.pad(events)：torch.Size([1, 5, 184, 240])
                    reconstructions_for_each_channel = {}
                    if self.perform_color_reconstruction:  #原代码版本未执行
                        events_for_each_channel['R'] = self.crop_halfres.pad(events[:, :, 0::2, 0::2])
                        events_for_each_channel['G'] = self.crop_halfres.pad(events[:, :, 0::2, 1::2])
                        events_for_each_channel['W'] = self.crop_halfres.pad(events[:, :, 1::2, 0::2])
                        events_for_each_channel['B'] = self.crop_halfres.pad(events[:, :, 1::2, 1::2])
                        
                    # 此处已调用模型进行训练，建立新图
                    # Reconstruct new intensity image for each channel (grayscale + RGBW if color reconstruction is enabled)
                    for channel in events_for_each_channel.keys():
                        #对于auto_hdr类型的运行代码来说，只有channel：graysca
                        #print('states before trained by model:',self.last_states_for_each_channel[channel]) #答案为None
                        with CudaTimer('Inference'):
                            new_predicted_frame, states = self.model(events_for_each_channel[channel],
                                                                        self.last_states_for_each_channel[channel])
                            #new_predicted_frame torch.Size([1, 1, 184, 240])
                            #states是list,type(states)=<class 'list'>
                            #states里的元素：<generator object ImageReconstructor.update_reconstruction.<locals>.<genexpr> at 0x7fa8d32eec10>
                        if self.no_recurrent:
                            self.last_states_for_each_channel[channel] = None
                        else:
                            #运行的这个
                            self.last_states_for_each_channel[channel] = states
                            #print('states\'type after trained by model:',type(states))
                            
                        # Output reconstructed image
                        crop = self.crop if channel == 'grayscale' else self.crop_halfres

                        # Unsharp mask (on GPU)
                        new_predicted_frame = self.unsharp_mask_filter(new_predicted_frame)

                        
                        # Intensity rescaler (on GPU)
                        new_predicted_frame = self.intensity_rescaler(new_predicted_frame)


                        
                        with Timer('Tensor (GPU) -> NumPy (CPU)'):
                            reconstructions_for_each_channel[channel] = new_predicted_frame[0, 0, crop.iy0:crop.iy1,
                                                                                            crop.ix0:crop.ix1]# .cpu().numpy() 去掉这，保持torch
                    if self.perform_color_reconstruction:
                        out = merge_channels_into_color_image(reconstructions_for_each_channel)  #没用到，暂时没改里面的np
                    else:
                        out = reconstructions_for_each_channel['grayscale']
        # Post-processing, e.g bilateral filter (on CPU)  #此处暂时改成了gpu版本
        out = self.image_filter(out)
        if self.train_model:
            return out
        else:
            # torch.set_printoptions(profile="full")
            # print(out)
            # print(out.shape)
            out = np.array(out.cpu())

            #stamp每次是拿每个窗口的last_timestamp来进行操作的
            self.image_writer(out, event_tensor_id, stamp, events=events)
            self.image_display(out, events)
