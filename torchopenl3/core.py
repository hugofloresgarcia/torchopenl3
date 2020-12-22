import numpy as np
import torch

SAMPLE_RATE = 48000
INPUT_REPRESENTATIONS = ('mel128', 'mel256')
CONTENT_TYPES = ('music', 'env')

BASE_WEIGHT_PATH = 'assets/weights'

def _get_weight_path(content_type, input_repr):
    return os.path.join(BASE_WEIGHT_PATH, f'{content_type}-{input_repr}')

def _check_embedding_size(embedding_size):
    assert isinstance(embedding_size, int), f'embedding_size must be type int'
    if embedding_size == 512:
        maxpool_kernel=(16, 24)
    elif embedding_size == 6144:
        maxpool_kernel=(4, 8)
    else: 
        raise ValueError(f'embedding_size should be 512 or 6144 but got {embedding_size}')

def _check_content_type(content_type):
     # check content types
    assert isinstance(content_type, str)
    assert content_type in CONTENT_TYPES, 
            f'content_type must be one of {CONTENT_TYPES}'

def _check_input_repr(input_repr):
        # check input representation
    assert isinstance(input_repr, str)
    assert input_repr in INPUT_REPRESENTATIONS, 
            f'input_repr must be one of {INPUT_REPRESENTATIONS}'

def _load_spectrogram_model(input_repr):
    if input_repr == 'mel128':
        return Melspectrogram(sr=SAMPLE_RATE, n_mels=128,
                              fmin=0.0, fmax=None, power_melgram=1.0, 
                              return_decibel_melgram=True, trainable_fb=False, 
                              htk=True)
    elif input_repr == 'mel256':
        return Melspectrogram(sr=SAMPLE_RATE, n_mels=256,
                              fmin=0.0, fmax=None, power_melgram=1.0, 
                              return_decibel_melgram=True, trainable_fb=False, 
                              htk=True)

def _load_openl3_model(input_repr, embedding_size, content_type, pretrained):
    weight_path = _get_weight_path(content_type, input_repr)

    return OpenL3Mel128(weight_path=weight_path, maxpool_kernel=maxpool_kernel,  
                        pretrained=pretrained)

class OpenL3Mel128(pl.LightningModule):

    def __init__(self, 
                weight_path: str, 
                input_shape=(1, 48000), 
                maxpool_kernel=(16, 24), 
                maxpool_stride=(4, 8), 
                pretrained=True):
        super(OpenL3Mel128, self).__init__()

        self.pretrained = pretrained
        self._weights_dict = load_weights(weight_path)

        self.batch_normalization_1 = self.__batch_normalization(2, 'batch_normalization_1', num_features=1, eps=0.0010000000474974513, momentum=0.50, track_running_stats=True)
        self.conv2d_1 = self.__conv(2, name='conv2d_1', in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_2 = self.__batch_normalization(2, 'batch_normalization_2', num_features=64, eps=0.0010000000474974513, momentum=0.50, track_running_stats=True)
        self.conv2d_2 = self.__conv(2, name='conv2d_2', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_3 = self.__batch_normalization(2, 'batch_normalization_3', num_features=64, eps=0.0010000000474974513, momentum=0.50, track_running_stats=True)
        self.conv2d_3 = self.__conv(2, name='conv2d_3', in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_4 = self.__batch_normalization(2, 'batch_normalization_4', num_features=128, eps=0.0010000000474974513, momentum=0.50, track_running_stats=True)
        self.conv2d_4 = self.__conv(2, name='conv2d_4', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_5 = self.__batch_normalization(2, 'batch_normalization_5', num_features=128, eps=0.0010000000474974513, momentum=0.50, track_running_stats=True)
        self.conv2d_5 = self.__conv(2, name='conv2d_5', in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_6 = self.__batch_normalization(2, 'batch_normalization_6', num_features=256, eps=0.0010000000474974513, momentum=0.50, track_running_stats=True)
        self.conv2d_6 = self.__conv(2, name='conv2d_6', in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_7 = self.__batch_normalization(2, 'batch_normalization_7', num_features=256, eps=0.0010000000474974513, momentum=0.50, track_running_stats=True)
        self.conv2d_7 = self.__conv(2, name='conv2d_7', in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_8 = self.__batch_normalization(2, 'batch_normalization_8', num_features=512, eps=0.0010000000474974513, momentum=0.50, track_running_stats=True)
        self.audio_embedding_layer = self.__conv(2, name='audio_embedding_layer', in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel, stride=maxpool_stride, padding=0, ceil_mode=False)

    def forward(self, x):
        batch_normalization_1 = self.batch_normalization_1(x)
        conv2d_1_pad    = F.pad(batch_normalization_1, (1, 1, 1, 1))
        conv2d_1        = self.conv2d_1(conv2d_1_pad)
        batch_normalization_2 = self.batch_normalization_2(conv2d_1)
        activation_1    = F.relu(batch_normalization_2)
        conv2d_2_pad    = F.pad(activation_1, (1, 1, 1, 1))
        conv2d_2        = self.conv2d_2(conv2d_2_pad)
        batch_normalization_3 = self.batch_normalization_3(conv2d_2)
        activation_2    = F.relu(batch_normalization_3)
        max_pooling2d_1, max_pooling2d_1_idx = F.max_pool2d(activation_2, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        conv2d_3_pad    = F.pad(max_pooling2d_1, (1, 1, 1, 1))
        conv2d_3        = self.conv2d_3(conv2d_3_pad)
        batch_normalization_4 = self.batch_normalization_4(conv2d_3)
        activation_3    = F.relu(batch_normalization_4)
        conv2d_4_pad    = F.pad(activation_3, (1, 1, 1, 1))
        conv2d_4        = self.conv2d_4(conv2d_4_pad)
        batch_normalization_5 = self.batch_normalization_5(conv2d_4)
        activation_4    = F.relu(batch_normalization_5)
        max_pooling2d_2, max_pooling2d_2_idx = F.max_pool2d(activation_4, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        conv2d_5_pad    = F.pad(max_pooling2d_2, (1, 1, 1, 1))
        conv2d_5        = self.conv2d_5(conv2d_5_pad)
        batch_normalization_6 = self.batch_normalization_6(conv2d_5)
        activation_5    = F.relu(batch_normalization_6)
        conv2d_6_pad    = F.pad(activation_5, (1, 1, 1, 1))
        conv2d_6        = self.conv2d_6(conv2d_6_pad)
        batch_normalization_7 = self.batch_normalization_7(conv2d_6)
        activation_6    = F.relu(batch_normalization_7)
        max_pooling2d_3, max_pooling2d_3_idx = F.max_pool2d(activation_6, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        conv2d_7_pad    = F.pad(max_pooling2d_3, (1, 1, 1, 1))
        conv2d_7        = self.conv2d_7(conv2d_7_pad)
        batch_normalization_8 = self.batch_normalization_8(conv2d_7)
        activation_7    = F.relu(batch_normalization_8)
        audio_embedding_layer_pad = F.pad(activation_7, (1, 1, 1, 1))
        audio_embedding_layer = self.audio_embedding_layer(audio_embedding_layer_pad)
        max_pooling2d_4 =  self.maxpool(audio_embedding_layer)
        # flatten_1       = max_pooling2d_4.view(max_pooling2d_4.size(0), -1)
            
        return max_pooling2d_4

    def __conv(self, dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        if self.pretrained:
            layer.state_dict()['weight'].copy_(torch.from_numpy(self._weights_dict[name]['weights']))
            if 'bias' in self._weights_dict[name]:
                layer.state_dict()['bias'].copy_(torch.from_numpy(self._weights_dict[name]['bias']))
        return layer

    def __batch_normalization(self, dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        if self.pretrained:
            if 'scale' in self._weights_dict[name]:
                layer.state_dict()['weight'].copy_(torch.from_numpy(self._weights_dict[name]['scale']))
            else:
                layer.weight.data.fill_(1)

            if 'bias' in self._weights_dict[name]:
                layer.state_dict()['bias'].copy_(torch.from_numpy(self._weights_dict[name]['bias']))
            else:
                layer.bias.data.fill_(0)

            layer.state_dict()['running_mean'].copy_(torch.from_numpy(self._weights_dict[name]['mean']))
            layer.state_dict()['running_var'].copy_(torch.from_numpy(self._weights_dict[name]['var']))
        return layer