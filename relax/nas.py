import pdb
from copy import deepcopy
import torch
from torch import nn, optim
from torch._six import inf
from relax.ops import Conv, int2tuple
from relax.xd import XD


def get_module(model, module_string):

    if module_string:
        for substring in module_string.split('.'):
            model = getattr(model, substring)
    return model


def check_weight_norm(module):

    for key, value in module._forward_pre_hooks.items():
        if type(value) == type(list(nn.utils.weight_norm(Conv(1)(1,1,1))._forward_pre_hooks.items())[0][1]):
            return value


class Supernet(nn.Sequential):

    @classmethod
    def create(cls, model, in_place=False, attrs=[]):
        '''
        Args:
            model: backbone model
            in_place: replace backbone model layers in place
            attrs: custom attributes of model to replace
        '''
        
        model = model if in_place else deepcopy(model)
        attrs = attrs if attrs else dir(model)
        assert 'forward' in attrs, "if nonempty, 'attrs' must contain 'forward'"
        attrs = [(attr, getattr(model, attr)) for attr in attrs if not attr[:2] == '__']
        model.__class__ = cls
        for name, attr in attrs:
            setattr(model, name, attr)
        return model

    def named_arch_params(self):
        '''iterates through (name, param) pairs of all architecture parameters in the model'''

        for name, module in self.named_modules():
            if name and hasattr(module, 'named_arch_params'):
                for n, p in module.named_arch_params():
                    yield name + '.' + n, p

    def arch_params(self):
        '''iterates through all architecture parameters in the model'''

        return (p for _, p in self.named_arch_params())

    def named_model_weights(self):
        '''iterates through (name, param) pairs of all model weights in the model'''

        exclude = {name for name, _ in self.named_arch_params()}
        return ((n, p) for n, p in self.named_parameters() if not n in exclude)

    def model_weights(self):
        '''iterates through all model weights in the model'''

        return (p for _, p in self.named_model_weights())

    def patch2xd(self, module_string, sample_input, sample_output, *args, test=False, test_boundary=0, func=lambda m: m, **kwargs):
        '''patches specified module with a XD
        Args:
            module_string: name of module to replace
            sample_input: sample input into the module.forward function
            sample_output: sample output of the module.forward function
            args: passed to xd.XD
            test: test agreement of replacement module on 'sample_input' and return relative error
            test_boundary: sets boundary when testing replacement module
            func: function to apply to xd.XD object before patching
            kwargs: passed to xd.XD
        '''

        if sample_input is None:
            module, test = None, False
        else:
            in_size = sample_input.shape[2:]
            in_channels = sample_input.shape[1]
            out_channels = sample_output.shape[1]
            module = func(XD(in_channels, out_channels, *args, in_size, odd=in_size[-1] % 2, **kwargs))

        while True:
            module_split = module_string.split('.')
            parent = get_module(self, '.'.join(module_split[:-1]))
            name = module_split[-1]
            child = getattr(parent, name)
            setattr(parent, module_split[-1], module)
            for module_string, m in self.named_modules():
                if m == child:
                    break
            else:
                break

        if test:
            test_boundary = int2tuple(test_boundary, length=len(in_size))
            slc = [slice(None), slice(None)] + [slice(b, n-b) for b, n in zip(int2tuple(test_boundary, length=len(in_size)), in_size)]
            output = module(sample_input)
            return module, (torch.norm(output[slc] - sample_output[slc]) / torch.norm(sample_output[slc])).item()
        return module, "module not used in forward pass" if sample_input is None else None

    def collect_io(self, sample_input, modules, *args):

        module_io = {}
        handles = [m.register_forward_hook(lambda s, i, o: module_io.__setitem__(s, (i[0], o))) for m in modules]
        self(sample_input, *args)
        for handle in handles:
            handle.remove()
        return module_io

    def conv2xd(self, sample_input, *args, named_modules=None, warm_start=True, verbose=False, padding='auto', padding_mode='auto', **kwargs):
        '''
        Args:
            sample_input: torch.Tensor of shape [batch-size, input-channels, *input-width]
            args: additional arguments passed to self.forward
            named_modules: iterable of named modules ; if None uses all modules in self.model
            warm_start: whether to initialize modules as 2d convs
            verbose: print patch logs
            padding: if 'auto' uses padding from target module
            paddin_mode: if 'auto' uses padding_mode from target module
            kwargs: passed to self.patch
        '''

        named_modules = self.named_modules() if named_modules is None else named_modules
        named_modules = [(n, m) for n, m in named_modules if hasattr(m, 'kernel_size') and type(m.kernel_size) == tuple and type(m) == Conv(len(m.kernel_size))]
        module_io = self.collect_io(sample_input, (m for _, m in named_modules), *args)

        for name, module in named_modules:
            ks = module.kernel_size
            arch_init = 'conv_' + 'x'.join(str(k) for k in ks)
            wn = check_weight_norm(module)
            msg = ""
            if wn is None:
                func = lambda m: m
            else:
                msg += "\tweight-norm detected"
                def func(m):
                    m = torch.nn.utils.weight_norm(m, dim=wn.dim)
                    if m.weight_g.shape == module.weight_g.shape:
                        m.weight_g = module.weight_g
                        m.weight_v = module.weight_v
                    return m
            m, err = self.patch2xd(name,
                                   *module_io.get(module, (None, None)),
                                   module.kernel_size,
                                   test=verbose,
                                   test_boundary=1,
                                   func=func,
                                   padding=module.padding if padding == 'auto' else padding,
                                   padding_mode=module.padding_mode if padding == 'auto' else padding_mode,
                                   stride=module.stride,
                                   dilation=module.dilation,
                                   groups=module.groups,
                                   weight=nn.Parameter(module.weight.data),
                                   bias=None if module.bias is None else nn.Parameter(module.bias.data),
                                   **kwargs)

            if verbose:
                print(name, '\terror:', err, msg)

    def save_arch(self, path):
        '''saves architecture parameters to provided filepath'''

        torch.save(dict(self.named_arch_params()), path)

    def load_arch(self, path, verbose=False):
        '''loads architecture parameters from provided filepath'''

        data = torch.load(path)
        for n, p in self.named_arch_params():
            load = data[n].data
            if p.data.shape == load.shape:
                p.data = load.to(p.device)
            elif verbose:
                print('did not load', n, '(shape mismatch)')

    def set_arch_requires_grad(self, requires_grad):
        '''sets 'requires_grad' attribute of architecture parameters to given value'''

        for param in self.arch_params():
            param.requires_grad = bool(requires_grad)


class MixedOptimizer(optim.Optimizer):

    def __init__(self, optimizers, alternating=False):
        '''
        Args:
            optimizers: list of objects that are subclasses of optim.Optimizer
            alternating: whether to alternate steps with different optimizers
        '''

        self.optimizers = []
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group['method'] = type(optimizer)
                group['initial_lr'] = group.get('initial_lr', group['lr'])
            self.optimizers.append(optimizer)
        super(MixedOptimizer, self).__init__((g for o in self.optimizers for g in o.param_groups), {})
        self.alternating = alternating
        self.iteration = 0

    def step(self, closure=None):

        if self.alternating:
            self.optimizers[self.iteration % len(self.optimizers)].step(closure=closure)
        else:
            for optimizer in self.optimizers:
                optimizer.step(closure=closure)
        self.iteration += 1


class MixedScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, scheduler):
        '''
        Args:
            optimizer: MixedOptimizer object
            schedulers: list of optim.lr_scheduler._LRScheduler objects
        '''

        self.schedulers = schedulers
        super(MixedScheduler, self).__init__(optimizer)

    def step(self, epoch=None):

        for scheduler in self.schedulers:
            scheduler.step()


def iter_grad(parameters):

    for param in parameters:
        try:
            yield param.grad.real.detach()
            yield param.grad.imag.detach()
        except RuntimeError:
            yield param.grad.detach()

def clip_grad_norm(parameters, max_norm, norm_type=2.0):
    '''handles gradient clipping for complex parameters'''

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(g.abs().max().to(device) for g in iter_grad(parameters))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g, norm_type).to(device) for g in iter_grad(parameters)]), norm_type)
    clip_coef = max_norm / (total_norm + 1E-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm
