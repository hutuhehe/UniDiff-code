"""
Helpers to train with 16-bit precision.
"""
# modify this part for lora fintuning or only label_emb
import numpy as np
import torch as th
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from . import logger
import pdb

INITIAL_LOG_LOSS_SCALE = 20.0

def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()
 
    if hasattr(l, 'lora_A') and hasattr(l, 'lora_B'):
        l.lora_A.data = l.lora_A.data.half()
        l.lora_B.data = l.lora_B.data.half()
        



def convert_module_to_f32(l): 
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()
    # Handle LoRA layers (assuming LoRA layers have attributes lora_A and lora_B)
    if hasattr(l, 'lora_A') and hasattr(l, 'lora_B'):
        l.lora_A.data = l.lora_A.data.float()
        l.lora_B.data = l.lora_B.data.float()


def make_master_params(param_groups_and_shapes):
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.
    """
  
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        master_param = nn.Parameter(
            _flatten_dense_tensors(
                [param.detach().float() for (_, param) in param_group]
            ).view(shape)
        )
        master_param.requires_grad = True
        master_params.append(master_param)
    return master_params


def model_grads_to_master_grads(param_groups_and_shapes, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    for master_param, (param_group, shape) in zip(
        master_params, param_groups_and_shapes
    ):
        master_param.grad = _flatten_dense_tensors(
            [param_grad_or_zeros(param) for (_, param) in param_group]
        ).view(shape)


def master_params_to_model_params(param_groups_and_shapes, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        for (_, param), unflat_master_param in zip(
            param_group, unflatten_master_params(param_group, master_param.view(-1))
        ):
            param.detach().copy_(unflat_master_param)


def unflatten_master_params(param_group, master_param):
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group])


def get_param_groups_and_shapes(named_model_params):
    
    named_model_params = list(named_model_params)
    scalar_vector_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim <= 1],
        (-1),
    )
    matrix_named_params = (
        [(n, p) for (n, p) in named_model_params if p.ndim > 1],
        (1, -1),
    )
    return [scalar_vector_named_params, matrix_named_params]


def master_params_to_state_dict(
    model, param_groups_and_shapes, master_params, use_fp16
):
    if use_fp16:
        state_dict = model.state_dict()
        for master_param, (param_group, _) in zip(
            master_params, param_groups_and_shapes
        ):
            for (name, _), unflat_master_param in zip(
                param_group, unflatten_master_params(param_group, master_param.view(-1))
            ):
                assert name in state_dict
                state_dict[name] = unflat_master_param
    else:
        state_dict = model.state_dict()
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
    return state_dict


def state_dict_to_master_params(model, state_dict, use_fp16):
    if use_fp16:
        named_model_params = [
            (name, state_dict[name]) for name, _ in model.named_parameters()
        ]
        param_groups_and_shapes = get_param_groups_and_shapes(named_model_params)
        master_params = make_master_params(param_groups_and_shapes)
    else:
        master_params = [state_dict[name] for name, _ in model.named_parameters()]
    return master_params


def zero_master_grads(master_params):
    for param in master_params:
        param.grad = None


def zero_grad(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


def param_grad_or_zeros(param):
    if param.grad is not None:
        return param.grad.data.detach()
    else:
        return th.zeros_like(param)


class MixedPrecisionTrainer:
    def __init__(
        self,
        *,
        model,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
    ):
        self.model = model 
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.param_groups_and_shapes = None
        self.lg_loss_scale = initial_lg_loss_scale

        if self.use_fp16:
            self.param_groups_and_shapes = get_param_groups_and_shapes(
                self.model.named_parameters()
            )
            self.master_params = make_master_params(self.param_groups_and_shapes)
            self.model.convert_to_fp16()

    def zero_grad(self):
        zero_grad(self.model_params)

    def backward(self, loss: th.Tensor):
        if self.use_fp16:
     
            loss_scale = 2 ** self.lg_loss_scale
            
            """
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(f"Name: {name}, Shape: {param.shape}")
            """
     
            
           # Ensure all parameters are trainable,error run out without this
           # for param in self.model.parameters():
            #param.requires_grad = True
     
            (loss * loss_scale).backward()

            
        else:
            loss.backward()

    def optimize(self, opt: th.optim.Optimizer):
        if self.use_fp16:
            return self._optimize_fp16(opt)
        else:
            return self._optimize_normal(opt)

    def _optimize_fp16(self, opt: th.optim.Optimizer):
        logger.logkv_mean("lg_loss_scale", self.lg_loss_scale)
        # update only lora part ,make the other part gradient zero

        # Iterate through model parameters
        """
        # modify for lora fintune 
        for name, param in self.model.named_parameters():
        # Check if the parameter is a LoRA parameter and has a gradient
            if 'lora_' in name and param.grad is not None:
                print(f"LoRA Parameter: {name}, Gradient: {param.grad}")
        """
        

        # Zero out gradients for non-LoRA and "label_emb"  parameters before copying
        # only make emb_layer trainable
        """
        for name, param in self.model.named_parameters():
            # pdb.set_trace()
            #if 'label_emb' not in name:  # Check if the parameter is not a LoRA paramete
            if 'emb_layers' not in name:
                if param.grad is not None:
                    param.grad.zero_()    # Zero out gradients
        """
        """

        # make emb_layer and label_emb both trainable, zero out other parameter modify on Feb 16th
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param.grad.zero_()  
        """
        """        
        # make emb_layer and label_emb both trainable, zero out other parameter modify on Feb 16th
        for name, param in self.model.named_parameters():
            if  'label_emb' in name : 
                continue
            if param.grad is not None:
                param.grad.zero_()  
        
       """
        # make emb_layer and label_emb both trainable, zero out other parameter modify on Feb 16th
        ####### modification on July 24 replace with requre grad true  and pair trainable with master params ###########
        '''
        for name, param in self.model.named_parameters():
            if 'emb_layers'  in name  or 'label_emb' in name or 'time_embed' in name: 
                continue
            if param.grad is not None:
                param.grad.zero_()  
        
        '''
        '''
        for name, param in self.model.named_parameters():
            if not param.requires_grad and param.grad is not None:
                param.grad.zero_()
        '''

       ## since have no better idea execpet this 
        for name, param in self.model.named_parameters():
            if "label_emb" in name or ("output_blocks" in name and "emb_layers" in name):
                continue  # keep grads
            if param.grad is not None:
                param.grad.zero_()  # zero grads for all other layers

        model_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)
        grad_norm, param_norm = self._compute_norms(grad_scale=2 ** self.lg_loss_scale)
        if check_overflow(grad_norm):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            zero_master_grads(self.master_params)
            return False
        
        logger.logkv_mean("grad_norm", grad_norm) 
        logger.logkv_mean("param_norm", param_norm)


        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        opt.step()
        zero_master_grads(self.master_params)
        master_params_to_model_params(self.param_groups_and_shapes, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth
        return True

    def _optimize_normal(self, opt: th.optim.Optimizer):
        grad_norm, param_norm = self._compute_norms()
        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)
        opt.step()
        return True

    # modify on July 24 filter param_norm computate realed to only trainable
    def _compute_norms(self, grad_scale=1.0):
        grad_norm = 0.0
        param_norm = 0.0
    
        with th.no_grad():
            for name, param in self.model.named_parameters():
                # ✅ Only include label_emb or output_blocks containing emb_layers
                if not ("label_emb" in name or ("output_blocks" in name and "emb_layers" in name)):
                    continue
    
                # Compute parameter L2 norm
                param_norm += th.norm(param, p=2, dtype=th.float32).item() ** 2
    
                # Compute gradient L2 norm if available
                if param.grad is not None:
                    grad_norm += th.norm(param.grad.float(), p=2).item() ** 2
    
        # Convert sums of squares into actual norms
        return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)




    '''
    def _compute_norms(self, grad_scale=1.0):
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.master_params:
            with th.no_grad():
                param_norm += th.norm(p, p=2, dtype=th.float32).item() ** 2
                if p.grad is not None:
                    grad_norm += th.norm(p.grad, p=2, dtype=th.float32).item() ** 2       
        return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)
    '''
    def master_params_to_state_dict(self, master_params):
        return master_params_to_state_dict(
            self.model, self.param_groups_and_shapes, master_params, self.use_fp16
        )

    def state_dict_to_master_params(self, state_dict):
        return state_dict_to_master_params(self.model, state_dict, self.use_fp16)


def check_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)
