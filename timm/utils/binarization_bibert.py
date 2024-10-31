from torch import nn as nn
import torch
import torch.nn.functional as F
from timm.layers.helpers import to_2tuple
from functools import partial

class BinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # out = torch.sign(input) + (input == 0).int()
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input
    
class QuantizeLinear(nn.Linear):
    def __init__(self,  *kargs,bias=True, config=None, type=None):
        super(QuantizeLinear, self).__init__(*kargs,bias=True)
        self.weight_quantizer = BinaryQuantizer()
        # self.register_buffer('weight_clip_val', torch.tensor([-1, 1]))
        self.init = True
            
        self.act_quantizer = BinaryQuantizer()
        # self.register_buffer('act_clip_val', torch.tensor([-1, 1]))
        # self.register_parameter('scale', torch.nn.parameter.Parameter(torch.Tensor([0.0]).squeeze()))
 
    def reset_scale(self, input):
        bw = self.weight
        ba = input
        self.scale = torch.nn.parameter.Parameter((ba.norm() / torch.sign(ba).norm()).float().to(ba.device))

    def forward(self, input, type=None):
        scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
        scaling_factor = scaling_factor.detach()
        real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights

        binary_input_no_grad = torch.sign(input)
        cliped_input = torch.clamp(input, -1.0, 1.0)
        ba = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input
        
        out = nn.functional.linear(ba, weight, self.bias)

        return out
    
    
class LearnableBias(nn.Module):
    def __init__(self, head_num, out_chn, val=0.):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, head_num, out_chn,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out
    
class ZMeanBinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        out[out==-1] = 0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input
    
class Mlp1w1a(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            bn_num=1,
            dense_knowledge=[],
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = QuantizeLinear(in_features, hidden_features, bias=bias[0])
        # self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        # self.fc2 = QuantizeLinear(hidden_features, out_features, bias=bias[1])
        
        self.moveact1 = LearnableBias2(bn_num)
        self.prelu = nn.PReLU(bn_num)
        self.moveact2 = LearnableBias2(bn_num)
        
        # self.fc2 = QuantizeLinear(197, 197, bias=bias[1])
        self.fc2 = QuantizeLinear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        
        self.movefc1 = LearnableBias2(bn_num)
        self.movefc2 = LearnableBias2(bn_num)
        
        self.dense_knowledge = dense_knowledge
        
        # kernel_size = 3
        # self.avg_res_w3 = nn.AvgPool2d((1, kernel_size), stride=1, padding=(0, int((kernel_size-1)/2)))
        # self.avg_res_h3 = nn.AvgPool2d((kernel_size, 1), stride=1, padding=(int((kernel_size-1)/2), 0))
        
        # kernel_size = 5
        # self.avg_res_w5 = nn.AvgPool2d((1, kernel_size), stride=1, padding=(0, int((kernel_size-1)/2)))
        # self.avg_res_h5 = nn.AvgPool2d((kernel_size, 1), stride=1, padding=(int((kernel_size-1)/2), 0))
        
        
        # self.bn = nn.BatchNorm1d(bn_num)
        # self.bn2 = nn.BatchNorm1d(bn_num)

    def forward(self, x):
        
        x = self.movefc1(x)
        x = self.fc1(x)
        # self.dense_knowledge[2].append(x)
        x = self.moveact1(x)
        x = self.prelu(x)
        x = self.moveact2(x)
        x = self.drop1(x)
        
        # x = self.bn(x)
        
        x = self.movefc2(x)
        x = self.fc2(x)
        # self.dense_knowledge[3].append(x)
        x = self.drop2(x)
        return x
    
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from torch.autograd import Function
import torch


class IRConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.rsign = nn.Parameter(torch.ones(3, 1, 1) / 2, requires_grad=True)

    def forward(self, input):
        w = self.weight
        input = input + self.rsign.expand_as(input)
        a = input
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(torch.tensor([2]*bw.size(0)).cuda().float(), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(bw.size(0), 1, 1, 1).detach()
        bw = BinaryQuantizer().apply(bw)
        ba = BinaryQuantizer().apply(a)
        bw = bw * sw
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output


from ..layers.format import Format, nchw_to
from ..layers.trace_utils import _assert
from typing import List, Optional, Callable

class PatchEmbed1w1a(nn.Module):
    """ 2D Image to Patch Embedding
    """
    output_fmt: Format

    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW

        self.proj = HardBinaryConv(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size).cuda()
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x



class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        #out_e1 = (x^2 + 2*x)
        #out_e2 = (-x^2 + 2*x)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out



class LearnableBias2(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias2, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size[0] * kernel_size[1]
        self.shape = (out_chn, in_chn, kernel_size[0], kernel_size[1])
        #self.weight = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)
        
        
        self.move0 = LearnableBias2(in_chn)
        self.binary_activation = BinaryActivation()
        # self.bn1 = nn.BatchNorm2d(out_chn)

    def forward(self, x):
        
        x = self.move0(x)
        x = self.binary_activation(x)
        
        #real_weights = self.weights.view(self.shape)
        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        #print(binary_weights, flush=True)
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        # y = self.bn1(y)
        return y
    
    

from torchvision.extension import _assert_has_ops
from typing import Optional, Tuple
from torch.nn.modules.utils import _pair
from torch.nn import init
from torch import Tensor

def deform_conv2d_tv(
    input: Tensor,
    offset: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    mask: Optional[Tensor] = None,
) -> Tensor:
    r"""
    Performs Deformable Convolution v2, described in
    `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168>`__ if :attr:`mask` is not ``None`` and
    Performs Deformable Convolution, described in
    `Deformable Convolutional Networks
    <https://arxiv.org/abs/1703.06211>`__ if :attr:`mask` is ``None``.

    Args:
        input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width]):
            offsets to be applied for each position in the convolution kernel.
        weight (Tensor[out_channels, in_channels // groups, kernel_height, kernel_width]): convolution weights,
            split into groups of size (in_channels // groups)
        bias (Tensor[out_channels]): optional bias of shape (out_channels,). Default: None
        stride (int or Tuple[int, int]): distance between convolution centers. Default: 1
        padding (int or Tuple[int, int]): height/width of padding of zeroes around
            each image. Default: 0
        dilation (int or Tuple[int, int]): the spacing between kernel elements. Default: 1
        mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width, out_height, out_width]):
            masks to be applied for each position in the convolution kernel. Default: None

    Returns:
        Tensor[batch_sz, out_channels, out_h, out_w]: result of convolution

    Examples::
        >>> input = torch.rand(4, 3, 10, 10)
        >>> kh, kw = 3, 3
        >>> weight = torch.rand(5, 3, kh, kw)
        >>> # offset and mask should have the same spatial size as the output
        >>> # of the convolution. In this case, for an input of 10, stride of 1
        >>> # and kernel size of 3, without padding, the output size is 8
        >>> offset = torch.rand(4, 2 * kh * kw, 8, 8)
        >>> mask = torch.rand(4, kh * kw, 8, 8)
        >>> out = deform_conv2d(input, offset, weight, mask=mask)
        >>> print(out.shape)
        >>> # returns
        >>>  torch.Size([4, 5, 8, 8])
    """
    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(deform_conv2d_tv)
    _assert_has_ops()
    out_channels = weight.shape[0]

    use_mask = mask is not None

    if mask is None:
        mask = torch.zeros((input.shape[0], 0), device=input.device, dtype=input.dtype)

    if bias is None:
        bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    weights_h, weights_w = weight.shape[-2:]
    _, n_in_channels, _, _ = input.shape

    n_offset_grps = offset.shape[1] // (2 * weights_h * weights_w)
    n_weight_grps = n_in_channels // weight.shape[1]

    if n_offset_grps == 0:
        raise RuntimeError(
            "the shape of the offset tensor at dimension 1 is not valid. It should "
            "be a multiple of 2 * weight.size[2] * weight.size[3].\n"
            f"Got offset.shape[1]={offset.shape[1]}, while 2 * weight.size[2] * weight.size[3]={2 * weights_h * weights_w}"
        )

    return torch.ops.torchvision.deform_conv2d(
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        n_weight_grps,
        n_offset_grps,
        use_mask,
    )

class CycleFC1w1a(nn.Module):
    """
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,  # re-defined kernel_size, represent the spatial area of staircase FC
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(CycleFC1w1a, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))  # kernel size == 1

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('offset', self.gen_offset())

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def gen_offset(self):
        """
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
            out_height, out_width]): offsets to be applied for each position in the
            convolution kernel.
        """
        offset = torch.empty(1, self.in_channels*2, 1, 1)
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1, self.kernel_size
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                offset[0, 2 * i + 1, 0, 0] = (i + start_idx) % self.kernel_size[1] - (self.kernel_size[1] // 2)
            else:
                offset[0, 2 * i + 0, 0, 0] = (i + start_idx) % self.kernel_size[0] - (self.kernel_size[0] // 2)
                offset[0, 2 * i + 1, 0, 0] = 0
        return offset

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        """
        B, C, H, W = input.size()
        input = BinaryQuantizer().apply(input)
        we = BinaryQuantizer().apply(self.weight)
        return deform_conv2d_tv(input, self.offset.expand(B, -1, H, W), we, self.bias, stride=self.stride,
                                padding=self.padding, dilation=self.dilation)

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)
    
    
class CycleMLP1w1a(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.mlp_c = QuantizeLinear(dim, dim, bias=qkv_bias)

        self.sfc_h = CycleFC1w1a(dim, dim, (1, 3), 1, 0)
        self.sfc_w = CycleFC1w1a(dim, dim, (3, 1), 1, 0)

        self.reweight = Mlp1w1a(dim, dim // 4, dim * 3)
        
        self.fc_spatial = QuantizeLinear(197, 197)

        # self.proj = QuantizeLinear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        cls_token = x[:, 0]
        
        x = x[:, 1:]
        x = x.reshape(-1, 14, 14, 192)
        B, H, W, C = x.shape
        
        h = self.sfc_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        w = self.sfc_w(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]
        
        x = x.reshape(B, -1, 192)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        
        x = x.transpose(1, 2)
        x = self.fc_spatial(x)
        x = x.transpose(1, 2)

        # x = self.proj(x)
        # x = self.proj_drop(x)

        return x
    
    
    
def att_loss_r2b(Q_s, Q_t):
    Q_s_norm = Q_s / torch.norm(Q_s, p=2)
    Q_t_norm = Q_t / torch.norm(Q_t, p=2)
    tmp = Q_s_norm - Q_t_norm
    loss = torch.norm(tmp, p=2)
    return loss

def direction_matching_distillation(student_scores, teacher_scores):
    tmp_loss = 0.
    for idx in range(len(student_scores)):
        # student_score = torch.matmul(student_scores[idx], student_scores[idx].transpose(-1, -2))
        # teacher_score = torch.matmul(teacher_scores[idx], teacher_scores[idx].transpose(-1, -2))
        
        student_score = student_scores[idx]
        teacher_score = teacher_scores[idx]
        
        student_score = torch.where(student_score <= -1e2, 
                                    torch.zeros_like(student_score).cuda(),
                                    student_score)
        teacher_score = torch.where(teacher_score <= -1e2,
                                    torch.zeros_like(teacher_score).cuda(),
                                    teacher_score)
        tmp_loss += att_loss_r2b(student_score, teacher_score)
    return tmp_loss

def cosine_distillation(student_scores, teacher_scores):
    tmp_loss = 0.
    for idx in range(len(student_scores)):
        student_score = student_scores[idx]
        teacher_score = teacher_scores[idx]
        student_score = student_score.permute(0, 2, 1)
        teacher_score = teacher_score.permute(0, 2, 1)
        student_score = torch.where(student_score <= -1e2, 
                                    torch.zeros_like(student_score).cuda(),
                                    student_score)
        teacher_score = torch.where(teacher_score <= -1e2,
                                    torch.zeros_like(teacher_score).cuda(),
                                    teacher_score)
        tmp_loss += (1 - F.cosine_similarity(student_score, teacher_score, dim=-1)).mean()
    return tmp_loss