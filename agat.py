import numpy as np
import os
import torch

def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x

class AGATProbe(object):
    def __init__(
        self,
        modules=None,
        output_file=None,
        enabled=False,
        append=False,
        iteration=0,
    ):
        assert modules is not None, "Must provide a valid torch model"
        assert output_file is not None, "Must provide an output file"

        self.enabled = enabled
        self.append = append
        self.iteration = iteration

        self.rank = torch.distributed.get_rank()

        self.output_file = f"{output_file}_{self.rank}"

        if not self.append:
            print(f"Starting new AGAT file {output_file}_{self.rank}\n")
            f = open(self.output_file, "w")
            f.close()
            for module in modules:
                self.parents = []
                self.get_children(module)
                for name, parameter in module.named_parameters():
                    parameter.register_hook(self.parameter_hook(name, parameter))
            self.append = True

    def parameter_hook(self, name, parameter):
        def f(grad):
            if grad is not None:
                key = f"grad_{name}_{self.rank}"
                nans = torch.isnan(grad).any()
                val = np.float64(grad.detach().double().abs().sum().item())
                with open(self.output_file, "a" if self.append else "w") as f:
                    f.write(f"{self.iteration},{key},{val},{int(nans)}\n")

        return f

    def get_children(self, model: torch.nn.Module):
        # get children form model!
        children = dict(model.named_children())
        if len(children.keys()) > 0:
            # look for children from children... to the last child!
            for name, module in children.items():
                module.register_forward_pre_hook(self.enter_module(name))
                module.register_forward_hook(self.exit_module(name))
                self.get_children(module)

    def enter_module(self, name):
        def f(module, inputs):
            self.parents.append(name)
            inputs = normalize_tuple(inputs)
            for inp_id, inp in enumerate(inputs):
                if inp is not None and isinstance(inp, torch.Tensor):
                    key = f"in_{name}_{inp_id}_{self.rank}"
                    nans = torch.isnan(inp).any()
                    val = float(inp.detach().double().abs().sum().item())
                    with open(self.output_file, "a" if self.append else "w") as f:
                        f.write(f"{self.iteration},{key},{val},{int(nans)}\n")

        return f

    def exit_module(self, name):
        def f(module, inputs, outputs):
            assert self.parents[-1] == name
            self.parents.pop()
            outputs = normalize_tuple(outputs)
            for out_id, out in enumerate(outputs):
                if out is not None and isinstance(out, torch.Tensor):
                    key = f"out_{name}_{out_id}_{self.rank}"
                    nans = torch.isnan(out).any()
                    val = float(out.detach().double().abs().sum().item())
                    with open(self.output_file, "a" if self.append else "w") as f:
                        f.write(f"{self.iteration},{key},{val},{int(nans)}\n")

        return f

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

