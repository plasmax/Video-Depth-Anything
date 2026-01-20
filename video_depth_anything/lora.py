import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=4):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        if rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            self.reset_parameters()

    def reset_parameters(self):
        if self.rank > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        if self.rank > 0:
            return (x @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return 0

class LinearWithLoRA(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=4):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features, 
            linear_layer.out_features, 
            rank, 
            alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

def inject_lora(model, rank=4, alpha=4, target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]):
    """
    Injects LoRA into the model by replacing targeted Linear layers with LinearWithLoRA.
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            inject_lora(module, rank, alpha, target_modules)
        
        # Check if this module matches target modules
        # This is a simplified check. You might want to pass the full name path if needed.
        full_name_match = any(target in name for target in target_modules) 
        
        # Ideally, we check the type and the name
        if isinstance(module, nn.Linear): # and check name if we want to be specific
             # For DinoV2, attention layers are usually key.
             # Let's replace ALL Linear layers inside blocks if we want, or specific ones.
             # The 'target_modules' arg suggests filtering.
             # However, since we are recursing, 'name' is just the child name (e.g. 'qkv').
             pass 

    # Re-implementation with better traversal
    replace_linear_with_lora(model, rank, alpha)

def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Check if we should replace this layer
            # For now, let's target attention projections and MLP layers in transformer blocks
            # Commonly: qkv, proj, fc1, fc2
            target_names = ['qkv', 'proj', 'fc1', 'fc2']
            if any(t in name for t in target_names):
                new_layer = LinearWithLoRA(module, rank, alpha)
                setattr(model, name, new_layer)
        else:
            replace_linear_with_lora(module, rank, alpha)

def mark_only_lora_as_trainable(model):
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
