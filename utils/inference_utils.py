from .util import robust_min, robust_max
from .path_utils import ensure_dir
from .timers import Timer, CudaTimer
from .loading_utils import get_device
from os.path import join
from math import ceil, floor
from torch.nn import ReflectionPad2d
import numpy as np
import torch
import cv2
from collections import deque
import atexit
import scipy.stats as st
import torch.nn.functional as F
from math import sqrt


def make_event_preview(events, mode='red-blue', num_bins_to_show=-1):
    # events: [1 x C x H x W] event tensor
    # mode: 'red-blue' or 'grayscale'
    # num_bins_to_show: number of bins of the voxel grid to show. -1 means show all bins.
    assert(mode in ['red-blue', 'grayscale'])
    if num_bins_to_show < 0:
        sum_events = torch.sum(events[0, :, :, :], dim=0).detach().cpu().numpy()
    else:
        sum_events = torch.sum(events[0, -num_bins_to_show:, :, :], dim=0).detach().cpu().numpy()

    if mode == 'red-blue':
        # Red-blue mode
        # positive events: blue, negative events: red
        event_preview = np.zeros((sum_events.shape[0], sum_events.shape[1], 3), dtype=np.uint8)
        b = event_preview[:, :, 0]
        r = event_preview[:, :, 2]
        b[sum_events > 0] = 255
        r[sum_events < 0] = 255
    else:
        # Grayscale mode
        # normalize event image to [0, 255] for display
        m, M = -10.0, 10.0
        event_preview = np.clip((255.0 * (sum_events - m) / (M - m)).astype(np.uint8), 0, 255)

    return event_preview


def gkern(kernlen=5, nsig=1.0):
    """Returns a 2D Gaussian kernel array."""
    """https://stackoverflow.com/a/29731818"""
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return torch.from_numpy(kernel).float()


class EventPreprocessor:
    """
    Utility class to preprocess event tensors.
    Can perform operations such as hot pixel removing, event tensor normalization,
    or flipping the event tensor.
        可以执行一些操作，如去除热像素，事件张量的正常化。或翻转事件张量。
    疑问：什么是去除热像素？
    """

    def __init__(self, options):
        if options.train_model == False:
            print('== Event preprocessing ==')
        self.no_normalize = options.no_normalize
        if self.no_normalize:
            print('!!Will not normalize event tensors!!')
        else:
            if options.train_model == False:
                print('Will normalize event tensors.')
        
        self.hot_pixel_locations = []
        if options.hot_pixels_file:
            try:
                self.hot_pixel_locations = np.loadtxt(options.hot_pixels_file, delimiter=',').astype(np.int)
                print('Will remove {} hot pixels'.format(self.hot_pixel_locations.shape[0]))
            except IOError:
                print('WARNING: could not load hot pixels file: {}'.format(options.hot_pixels_file))

        self.flip = options.flip
        if self.flip:
            print('Will flip event tensors.')

    def __call__(self, events):  #被调用了的，进行normalize

        # Remove (i.e. zero out) the hot pixels
        for x, y in self.hot_pixel_locations:
            events[:, :, y, x] = 0

        # Flip tensor vertically and horizontally
        if self.flip:
            events = torch.flip(events, dims=[2, 3])

        # Normalize the event tensor (voxel grid) so that
        # the mean and stddev of the nonzero values in the tensor are equal to (0.0, 1.0)
        if not self.no_normalize:
            with CudaTimer('Normalization'):
                nonzero_ev = (events != 0)  #过滤器,值为True，False

                num_nonzeros = nonzero_ev.sum()
                if num_nonzeros > 0:
                    # compute mean and stddev of the **nonzero** elements of the event tensor
                    # we do not use PyTorch's default mean() and std() functions since it's faster
                    # to compute it by hand than applying those funcs to a masked array
                    mean = events.sum() / num_nonzeros #直接将sum除以非零数
                    stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2) #标准差
                    mask = nonzero_ev.float()
                    #转为浮点数0，1了
                    events = mask * (events - mean) / stddev #此处为矩阵点乘

        return events


class IntensityRescaler:
    """
    Utility class to rescale image intensities to the range [0, 1],
    using (robust) min/max normalization.
    Optionally, the min/max bounds can be smoothed over a sliding window to avoid jitter.
    """

    def __init__(self, options):
        self.auto_hdr = options.auto_hdr
        self.intensity_bounds_min = torch.tensor(-99).reshape(1) # self.intensity_bounds=deque()为py的类list容器，现改为空tensor
        self.intensity_bounds_max = torch.tensor(-99).reshape(1) #仅有1维，特殊的初始化标识
        self.intensity_bounds=deque()
        self.auto_hdr_median_filter_size = options.auto_hdr_median_filter_size
        self.Imin = options.Imin
        self.Imax = options.Imax
        self.train_model = options.train_model
    def __call__(self, img):
        """
        param img: [1 x 1 x H x W] Tensor taking values in [0, 1]
        """
        if self.auto_hdr:
            #下面为了实现tensor连续训练，将np实现转为pytorch的
            with CudaTimer('Compute Imin/Imax (auto HDR)'):
                
                #原numpy的残余代码
                # # adjust image dynamic range (i.e. its contrast)
                #  #扔掉最开始的那个
                # if len(self.intensity_bounds) > self.auto_hdr_median_filter_size:
                #     self.intensity_bounds.popleft() 
                
                # self.intensity_bounds.append((Imin, Imax))
                # self.Imin = np.median([rmin for rmin, rmax in self.intensity_bounds]) #np实现
                # self.Imax = np.median([rmax for rmin, rmax in self.intensity_bounds])
                if self.train_model:
                    Imin = torch.min(img)#.item()  #此处将pytorch的转为了python的
                    Imax = torch.max(img)#.item()
                    
                    # ensure that the range is at least 0.1
                    Imin = torch.clamp(Imin, 0.0, 0.45)    #np.clip(Imin, 0.0, 0.45)
                    Imax = torch.clamp(Imax, 0.55, 1.0) # np.clip(Imax, 0.55, 1.0)
                    Imin = Imin.reshape(1) #新加，适应torch方法
                    Imax = Imax.reshape(1)
                    #数据大了的话，扔掉左边的
                    if self.intensity_bounds_min.shape[0] > self.auto_hdr_median_filter_size:
                        self.intensity_bounds_min = self.intensity_bounds_min[:,1:]
                        self.intensity_bounds_max = self.intensity_bounds_max[:,1:]
                    
                    if self.intensity_bounds_min[0] == -99:
                        self.intensity_bounds_min = Imin
                    else:
                        self.intensity_bounds_min = torch.cat(self.intensity_bounds_min,Imin)
                    if self.intensity_bounds_max[0] == -99:
                        self.intensity_bounds_max = Imax
                    else:
                        self.intensity_bounds_max = torch.cat(self.intensity_bounds_max,Imax)
                    self.Imin = torch.median(self.intensity_bounds_min)
                    self.Imax = torch.median(self.intensity_bounds_max)

                else:  #eval模式
                    Imin = torch.min(img).item()  #此处将pytorch的转为了python的
                    Imax = torch.max(img).item()
                    
                    # ensure that the range is at least 0.1
                    Imin = np.clip(Imin, 0.0, 0.45)
                    Imax = np.clip(Imax, 0.55, 1.0)
                    # adjust image dynamic range (i.e. its contrast)
                    if len(self.intensity_bounds) > self.auto_hdr_median_filter_size:
                        self.intensity_bounds.popleft() 
                    
                    self.intensity_bounds.append((Imin, Imax))
                    self.Imin = np.median([rmin for rmin, rmax in self.intensity_bounds]) #np实现
                    self.Imax = np.median([rmax for rmin, rmax in self.intensity_bounds])
        with CudaTimer('Intensity rescaling'):
            img = 255.0 * (img - self.Imin) / (self.Imax - self.Imin)
            img.clamp_(0.0, 255.0)
            if self.train_model == False:
                img = img.byte()  # convert to 8-bit tensor非训练模式
        return img


class ImageWriter:
    """
    Utility class to write images to disk.
    Also writes the image timestamps into a text file.
    """

    def __init__(self, options):

        self.output_folder = options.output_folder
        self.dataset_name = options.dataset_name
        self.save_events = options.show_events
        self.event_display_mode = options.event_display_mode
        self.num_bins_to_show = options.num_bins_to_show
        if options.train_model == False:
            print('== Image Writer ==')
            if self.output_folder:
                ensure_dir(self.output_folder)
                ensure_dir(join(self.output_folder, self.dataset_name))
                print('Will write images to: {}'.format(join(self.output_folder, self.dataset_name)))
                self.timestamps_file = open(join(self.output_folder, self.dataset_name, 'timestamps.txt'), 'a')

                if self.save_events:
                    self.event_previews_folder = join(self.output_folder, self.dataset_name, 'events')
                    ensure_dir(self.event_previews_folder)
                    print('Will write event previews to: {}'.format(self.event_previews_folder))

                atexit.register(self.__cleanup__)
            else:
                print('Will not write images to disk.')

    def __call__(self, img, event_tensor_id, stamp=None, events=None):
        if not self.output_folder:
            return

        if self.save_events and events is not None:
            event_preview = make_event_preview(events, mode=self.event_display_mode,
                                               num_bins_to_show=self.num_bins_to_show)
            cv2.imwrite(join(self.event_previews_folder,
                             'events_{:010d}.png'.format(event_tensor_id)), event_preview)

        cv2.imwrite(join(self.output_folder, self.dataset_name,
                         'frame_{:010d}.png'.format(event_tensor_id)), img)
        if stamp is not None:
            self.timestamps_file.write('{:.18f}\n'.format(stamp))

    def __cleanup__(self):
        if self.output_folder:
            self.timestamps_file.close()


class TrainModel:
    ''''
    我的训练模型部分的代码
    '''
    def __init__(self, options):

        self.output_folder = options.output_folder
        self.dataset_name = options.dataset_name
        self.save_events = options.show_events
        self.event_display_mode = options.event_display_mode
        self.num_bins_to_show = options.num_bins_to_show
        print('== Trian ==')
        if self.output_folder:
            ensure_dir(self.output_folder)
            ensure_dir(join(self.output_folder, self.dataset_name))
            print('Will write images to: {}'.format(join(self.output_folder, self.dataset_name)))
            self.timestamps_file = open(join(self.output_folder, self.dataset_name, 'timestamps.txt'), 'a')

            if self.save_events:
                self.event_previews_folder = join(self.output_folder, self.dataset_name, 'events')
                ensure_dir(self.event_previews_folder)
                print('Will write event previews to: {}'.format(self.event_previews_folder))

            atexit.register(self.__cleanup__)
        else:
            print('Will not write images to disk.')

    def __call__(self, img, event_tensor_id, stamp=None, events=None):
        if not self.output_folder:
            return

        if self.save_events and events is not None:
            event_preview = make_event_preview(events, mode=self.event_display_mode,
                                               num_bins_to_show=self.num_bins_to_show)
            cv2.imwrite(join(self.event_previews_folder,
                             'events_{:010d}.png'.format(event_tensor_id)), event_preview)

        cv2.imwrite(join(self.output_folder, self.dataset_name,
                         'frame_{:010d}.png'.format(event_tensor_id)), img)
        if stamp is not None:
            self.timestamps_file.write('{:.18f}\n'.format(stamp))

    def __cleanup__(self):
        if self.output_folder:
            self.timestamps_file.close()
class ImageDisplay:
    """
    Utility class to display image reconstructions
    """

    def __init__(self, options):
        self.display = options.display
        self.show_events = options.show_events
        self.color = options.color
        self.event_display_mode = options.event_display_mode
        self.num_bins_to_show = options.num_bins_to_show

        self.window_name = 'Reconstruction'
        if self.show_events:
            self.window_name = 'Events | ' + self.window_name

        if self.display:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.border = options.display_border_crop
        self.wait_time = options.display_wait_time

    def crop_outer_border(self, img, border):
        if self.border == 0:
            return img
        else:
            return img[border:-border, border:-border]

    def __call__(self, img, events=None):

        if not self.display:
            return

        img = self.crop_outer_border(img, self.border)

        if self.show_events:
            assert(events is not None)
            event_preview = make_event_preview(events, mode=self.event_display_mode,
                                               num_bins_to_show=self.num_bins_to_show)
            event_preview = self.crop_outer_border(event_preview, self.border)

        if self.show_events:
            img_is_color = (len(img.shape) == 3)
            preview_is_color = (len(event_preview.shape) == 3)

            if(preview_is_color and not img_is_color):
                img = np.dstack([img] * 3)
            elif(img_is_color and not preview_is_color):
                event_preview = np.dstack([event_preview] * 3)

            img = np.hstack([event_preview, img])

        cv2.imshow(self.window_name, img)
        cv2.waitKey(self.wait_time)


class UnsharpMaskFilter:
    """
    Utility class to perform unsharp mask filtering on reconstructed images.
    """

    def __init__(self, options, device):
        self.unsharp_mask_amount = options.unsharp_mask_amount
        self.unsharp_mask_sigma = options.unsharp_mask_sigma
        self.gaussian_kernel_size = 5
        self.gaussian_kernel = gkern(self.gaussian_kernel_size,
                                     self.unsharp_mask_sigma).unsqueeze(0).unsqueeze(0).to(device)

    def __call__(self, img):
        if self.unsharp_mask_amount > 0:
            with CudaTimer('Unsharp mask'):
                blurred = F.conv2d(img, self.gaussian_kernel,
                                   padding=self.gaussian_kernel_size // 2)
                img = (1 + self.unsharp_mask_amount) * img - self.unsharp_mask_amount * blurred
        return img


class ImageFilter:
    """
    Utility class to perform some basic filtering on reconstructed images.
    """

    def __init__(self, options):
        self.bilateral_filter_sigma = options.bilateral_filter_sigma

    def __call__(self, img):

        if self.bilateral_filter_sigma:
            with CudaTimer('Bilateral filter (sigma={:.2f})'.format(self.bilateral_filter_sigma)):
                #filtered_img = np.zeros_like(img)
                filtered_img = torch.zeros_like(img)
                filtered_img = cv2.bilateralFilter(
                    img, 5, 25.0 * self.bilateral_filter_sigma, 25.0 * self.bilateral_filter_sigma)
                img = filtered_img

        return img


def optimal_crop_size(max_size, max_subsample_factor):
    """ Find the optimal crop size for a given max_size and subsample_factor.
        The optimal crop size is the smallest integer which is greater or equal than max_size,
        while being divisible by 2^max_subsample_factor.
    """
    crop_size = int(pow(2, max_subsample_factor) * ceil(max_size / pow(2, max_subsample_factor)))
    return crop_size


class CropParameters:
    """ Helper class to compute and store useful parameters for pre-processing and post-processing
        of images in and out of E2VID.
        Pre-processing: finding the best image size for the network, and padding the input image with zeros
        Post-processing: Crop the output image back to the original image size
    """

    def __init__(self, width, height, num_encoders):

        self.height = height
        self.width = width
        self.num_encoders = num_encoders
        #width_crop_size就是要扩充到的面积
        self.width_crop_size = optimal_crop_size(self.width, num_encoders) #为何这个是最优？
        self.height_crop_size = optimal_crop_size(self.height, num_encoders)

        #padding_top就是依据扩充目标和扩前初始大小来确定的
        self.padding_top = ceil(0.5 * (self.height_crop_size - self.height))
        self.padding_bottom = floor(0.5 * (self.height_crop_size - self.height))
        self.padding_left = ceil(0.5 * (self.width_crop_size - self.width))
        self.padding_right = floor(0.5 * (self.width_crop_size - self.width))
        self.pad = ReflectionPad2d((self.padding_left, self.padding_right, self.padding_top, self.padding_bottom))

        self.cx = floor(self.width_crop_size / 2)
        self.cy = floor(self.height_crop_size / 2)

        self.ix0 = self.cx - floor(self.width / 2)
        self.ix1 = self.cx + ceil(self.width / 2)
        self.iy0 = self.cy - floor(self.height / 2)
        self.iy1 = self.cy + ceil(self.height / 2)


def shift_image(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy > 0:
        X[:dy, :] = np.expand_dims(X[dy, :], axis=0)
    elif dy < 0:
        X[dy:, :] = np.expand_dims(X[dy, :], axis=0)
    if dx > 0:
        X[:, :dx] = np.expand_dims(X[:, dx], axis=1)
    elif dx < 0:
        X[:, dx:] = np.expand_dims(X[:, dx], axis=1)
    return X


def upsample_color_image(grayscale_highres, color_lowres_bgr, colorspace='LAB'):
    """
    Generate a high res color image from a high res grayscale image, and a low res color image,
    using the trick described in:
    http://www.planetary.org/blogs/emily-lakdawalla/2013/04231204-image-processing-colorizing-images.html
    """
    assert(len(grayscale_highres.shape) == 2)
    assert(len(color_lowres_bgr.shape) == 3 and color_lowres_bgr.shape[2] == 3)

    if colorspace == 'LAB':
        # convert color image to LAB space
        lab = cv2.cvtColor(src=color_lowres_bgr, code=cv2.COLOR_BGR2LAB)
        # replace lightness channel with the highres image
        lab[:, :, 0] = grayscale_highres
        # convert back to BGR
        color_highres_bgr = cv2.cvtColor(src=lab, code=cv2.COLOR_LAB2BGR)
    elif colorspace == 'HSV':
        # convert color image to HSV space
        hsv = cv2.cvtColor(src=color_lowres_bgr, code=cv2.COLOR_BGR2HSV)
        # replace value channel with the highres image
        hsv[:, :, 2] = grayscale_highres
        # convert back to BGR
        color_highres_bgr = cv2.cvtColor(src=hsv, code=cv2.COLOR_HSV2BGR)
    elif colorspace == 'HLS':
        # convert color image to HLS space
        hls = cv2.cvtColor(src=color_lowres_bgr, code=cv2.COLOR_BGR2HLS)
        # replace lightness channel with the highres image
        hls[:, :, 1] = grayscale_highres
        # convert back to BGR
        color_highres_bgr = cv2.cvtColor(src=hls, code=cv2.COLOR_HLS2BGR)

    return color_highres_bgr


def merge_channels_into_color_image(channels):
    """
    Combine a full resolution grayscale reconstruction and four color channels at half resolution
    into a color image at full resolution.

    :param channels: dictionary containing the four color reconstructions (at quarter resolution),
                     and the full resolution grayscale reconstruction.
    :return a color image at full resolution
    """

    with CudaTimer('Merge color channels'):

        assert('R' in channels)
        assert('G' in channels)
        assert('W' in channels)
        assert('B' in channels)
        assert('grayscale' in channels)

        # upsample each channel independently
        for channel in ['R', 'G', 'W', 'B']:
            channels[channel] = cv2.resize(channels[channel], dsize=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        # Shift the channels so that they all have the same origin
        channels['B'] = shift_image(channels['B'], dx=1, dy=1)
        channels['G'] = shift_image(channels['G'], dx=1, dy=0)
        channels['W'] = shift_image(channels['W'], dx=0, dy=1)

        # reconstruct the color image at half the resolution using the reconstructed channels RGBW
        reconstruction_bgr = np.dstack([channels['B'],
                                        cv2.addWeighted(src1=channels['G'], alpha=0.5,
                                                        src2=channels['W'], beta=0.5,
                                                        gamma=0.0, dtype=cv2.CV_8U),
                                        channels['R']])

        reconstruction_grayscale = channels['grayscale']

        # combine the full res grayscale resolution with the low res to get a full res color image
        upsampled_img = upsample_color_image(reconstruction_grayscale, reconstruction_bgr)
        return upsampled_img

    return upsampled_img


def events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(np.int)
    ys = events[:, 2].astype(np.int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int)  #转换
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid

def hook_fn(grad):
    grad_file = open('nonleaf_grad.txt', 'a')
    grad_file.write('grad:{}\n'.format(grad))
def events_to_voxel_grid_pytorch(events, num_bins, width, height, device):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param device: device to use to perform computations
    :return voxel_grid: PyTorch event tensor (on the device specified)
    """

    DeviceTimer = CudaTimer if device.type == 'cuda' else Timer

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    # with torch.no_grad(): #禁用梯度计算，有利于减少内存消耗
    
    '''
    5.31
    备注：此时没有按train和image网络划分梯度运算，纯粹从训练模式的角度，且用的较隐晦，用的train模式下改变
    是在体素操作之前进行处理，而测试模式则无此改变。
    6.1:
    1. 注意到这里的运算十分复杂，不禁用梯度的话，会导致梯度运算报错
    2. 思考：以前这一部分本来也是不展开梯度运算的，因为只是一个普通的体素操作，我们在训练中最好保持
    3. 由于上下都有禁用梯度，可以将两者合并'''

    if not isinstance(events, np.ndarray):  #此时不是ndarray，即前面已经转化为tensor
        
        # events_torch = torch.from_numpy(events)
        events_torch = events
        #with torch.no_grad(): #为了训练，作为中间层的此处应该不能使用禁用梯度
        if 1:
            with DeviceTimer('Voxel grid voting'):
                voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device=device)
                voxel_grid = voxel_grid.flatten()
                
                # print('voxel_grid.requires_grad:',voxel_grid.requires_grad)
                #voxel_grid.requires_grad: False 新创建的叶节点，默认是属于false的
                # print('events_torch.requires_grad:',events_torch.requires_grad)
                # events_torch.requires_grad: True

                #print('voxel_grid is_leaf:',voxel_grid.is_leaf,'  voxel_grid.grad_fn:',voxel_grid.grad_fn)
                #voxel_grid is_leaf: True   voxel_grid.grad_fn: None

                # normalize the event timestamps so that they lie between 0 and num_bins
                last_stamp = events_torch[-1, 0]
                first_stamp = events_torch[0, 0]
                deltaT = last_stamp - first_stamp

                #print('events_torch is_leaf:',events_torch.is_leaf,'  events_torch.grad_fn:',events_torch.grad_fn)
                #events_torch is_leaf: False   events_torch.grad_fn: <CopySlices object at 0x7f37be55d6d0>

                if deltaT == 0: #防止遇上0导致错误
                    deltaT = 1.0

                events_torch[:, 0] = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
                ts = events_torch[:, 0]
                xs = events_torch[:, 1]
                ys = events_torch[:, 2]  #此时index为非叶节点，但是index_add无法加
                # xs = events_torch[:, 1].long()
                # ys = events_torch[:, 2].long() #此时index为叶子节点
                #.long()相关重写，看看能不能改为运算，前向反向进行自定义。

                #print('events_torch[:, 3].dtype',events_torch[:, 3].dtype)
                pols = events_torch[:, 3].float()  #此处转换类型没有影响的原因在于本来就是torch.float32在计算
                
                pols[pols == 0] = -1  # polarity should be +1 / -1

                tis = torch.floor(ts)  #向下取整
                tis_long = tis.long()  #转为Int64
                dts = ts - tis         #找到小数
                vals_left = pols * (1.0 - dts.float())
                vals_right = pols * dts.float()
                # print('vals_right is_leaf:',vals_right.is_leaf,'  vals_right.grad_fn:',vals_right.grad_fn)
                # vals_right is_leaf: False   vals_right.grad_fn: <MulBackward0 object at 0x7f9106d697f0>
                #此处说明了vals_right确实能回传grad的，而唯一与vals_right相关的只有pols和dts，那么这里就只能传回那两列

                valid_indices = tis < num_bins #原句
                valid_indices =  valid_indices & (tis >= 0)   #将tis>0的值(bool型)与valid_indices值进行按位与运算,防止inplace，则拆开
                #print('valid_indices is_leaf:',valid_indices.is_leaf,'  valid_indices.grad_fn:',valid_indices.grad_fn)
                #valid_indices is_leaf: True   valid_indices.grad_fn: None  valid_indices.grad:  None

                #dim=0,对进行index选取，进行加和
                # print('valid_indices len:',len(valid_indices),'  valid_indices:',valid_indices)
                # print('xs:',len(xs[valid_indices]))

                index=xs[valid_indices] + ys[valid_indices]* width + tis_long[valid_indices] * width * height
                print('index is_leaf:',index.is_leaf,'  index.grad_fn:',index.grad_fn)
                #index is_leaf: True   index.grad_fn: None

                source=vals_left[valid_indices]
                #print('source is_leaf:',source.is_leaf,'  source.grad_fn:',source.grad_fn)
                #source is_leaf: False   source.grad_fn: <IndexBackward0 object at 0x7fdfe5cf91c0>
                voxel_grid_1 = torch.index_add(voxel_grid,dim=0,
                                        index=xs[valid_indices] + ys[valid_indices]
                                        * width + tis_long[valid_indices] * width * height,
                                        source=vals_left[valid_indices])
                #这里voxel_grid、index都是叶节点，传到这儿直接终止，只有source不是，才能回传
                #初步构想的解决方案：index变成非叶，要与xs、ys扯上关系

                #voxel_grid_1.grad_fn: <IndexAddBackward0 object at 0x7f5bbd961850>
                #voxel_grid.shape = torch.Size([216000])
                valid_indices = (tis + 1) < num_bins
                valid_indices =  valid_indices & (tis >= 0)

                voxel_grid_2= torch.index_add(voxel_grid_1,dim=0,
                                        index=xs[valid_indices] + ys[valid_indices] * width
                                        + (tis_long[valid_indices] + 1) * width * height,
                                        source=vals_right[valid_indices])
                #voxel_grid_2.grad_fn: <IndexAddBackward0 object at 0x7f56ca0d7df0>
                #voxel_grid.shape = torch.Size([216000])

            voxel_grid_3 = voxel_grid_2.view(num_bins, height, width)
            #print('voxel_grid_3.grad_fn:',voxel_grid_3.grad_fn) #<ViewBackward0 object at 0x7ff58dbf4850>
            #voxel_grid_3.register_hook(hook_fn)
        #voxel_grid.shape=torch.Size([5, 180, 240])
    else: #若为ndarray
        with torch.no_grad(): #禁用梯度计算，有利于减少内存消耗
            events_torch = torch.from_numpy(events)
            with DeviceTimer('Events -> Device (voxel grid)'):
                events_torch = events_torch.to(device)

            with DeviceTimer('Voxel grid voting'):
                voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device=device).flatten()

                # normalize the event timestamps so that they lie between 0 and num_bins
                last_stamp = events_torch[-1, 0]
                first_stamp = events_torch[0, 0]
                deltaT = last_stamp - first_stamp

                if deltaT == 0: #防止遇上0导致错误
                    deltaT = 1.0

                events_torch[:, 0] = (num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
                ts = events_torch[:, 0]
                xs = events_torch[:, 1].long()
                ys = events_torch[:, 2].long()
                pols = events_torch[:, 3].float()
                pols[pols == 0] = -1  # polarity should be +1 / -1

                tis = torch.floor(ts)  #向下取整
                tis_long = tis.long()  #转为Int64
                dts = ts - tis         #找到小数
                vals_left = pols * (1.0 - dts.float())
                vals_right = pols * dts.float()

                valid_indices = tis < num_bins #原句
                valid_indices &= tis >= 0   #将tis>0的值(bool型)与valid_indices值进行按位与运算
                
                index=xs[valid_indices] + ys[valid_indices]* width + tis_long[valid_indices] * width * height
                torch.set_printoptions(profile="full")
                source=vals_left[valid_indices]
                
                # print('source shape:',source.shape)#' source',source)
                # print('index shape:',index.shape,' index',index)
                # exit(0)

                #没看懂这部分的内容，体元在操作什么？
                #dim=0,对进行index选取，进行加和
                voxel_grid.index_add_(dim=0,
                                        index=xs[valid_indices] + ys[valid_indices]
                                        * width + tis_long[valid_indices] * width * height,
                                        source=vals_left[valid_indices])
                #voxel_grid.shape = torch.Size([216000])
                valid_indices = (tis + 1) < num_bins
                valid_indices &= tis >= 0

                voxel_grid.index_add_(dim=0,
                                        index=xs[valid_indices] + ys[valid_indices] * width
                                        + (tis_long[valid_indices] + 1) * width * height,
                                        source=vals_right[valid_indices])
                #voxel_grid.shape = torch.Size([216000])
                
            voxel_grid_3 = voxel_grid.view(num_bins, height, width)
    return voxel_grid_3
