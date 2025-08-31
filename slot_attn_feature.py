''' Based on SLATE and BOQSA libraries:
https://github.com/singhgautam/slate/blob/master/slot_attn.py
https://github.com/YuLiu-LY/BO-QSA/blob/main/models/slot_attn.py
'''

from utils_spot import *
from timm.models.layers import DropPath

class SlotAttention(nn.Module):
    def __init__(
        self,
        num_iter,
        input_size,
        slot_size, 
        mlp_size, 
        truncate,
        heads,
        epsilon=1e-8, 
        drop_path=0,
    ):
        super().__init__()
        self.num_iter = num_iter
        self.input_size = input_size
        self.slot_size = slot_size
        self.epsilon = epsilon
        self.truncate = truncate
        self.num_heads = heads

        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)

        self.gru = gru_cell(slot_size, slot_size)

        self.mlp = nn.Sequential(
            linear(slot_size, mlp_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_size, slot_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        print(self.truncate)
        assert self.truncate in ['bi-level', 'fixed-point', 'none']


    def forward(self, inputs, slots_init):
        # `inputs` has shape [batch_size, num_inputs, input_size].
        # `slots` has shape [batch_size, num_slots, slot_size].
        slots = slots_init
        B, N_kv, D_inp = inputs.size()
        B, N_q, D_slot = slots.size()

        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs).view(B, N_kv, self.num_heads, -1).transpose(1, 2)    # Shape: [batch_size, num_heads, num_inputs, slot_size // num_heads].
        v = self.project_v(inputs).view(B, N_kv, self.num_heads, -1).transpose(1, 2)    # Shape: [batch_size, num_heads, num_inputs, slot_size // num_heads].
        
        
        batch_size,_,num_inputs,slots_size = k.shape
        H = W = int(num_inputs ** 0.5)
        k_1 = k.clone()
        # k_1 = k_1.squeeze(1).permute(0, 2, 1)
        k_1 = k_1.view(batch_size, slots_size, H, W)
        score = torch.einsum('bkd,bdhw->bkhw',F.normalize(slots,dim = 2),F.normalize(k_1,dim=1)) # shape [b,numslot,h,w]

        k = ((self.slot_size // self.num_heads) ** (-0.5)) * k
        # Multiple rounds of attention.
        for i in range(self.num_iter):
            if i == self.num_iter  - 1:
                if self.truncate == 'bi-level':
                    slots = slots.detach() + slots_init - slots_init.detach()
                elif self.truncate == 'fixed-point':
                    slots = slots.detach()
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # Attention.
            q = self.project_q(slots).view(B, N_q, self.num_heads, -1).transpose(1, 2)  # Shape: [batch_size, num_heads, num_slots, slot_size // num_heads].
            attn_logits = torch.matmul(k, q.transpose(-1, -2))                          # Shape: [batch_size, num_heads, num_inputs, num_slots].
            # q [128,1,7,256],k_1(128,256,14,14)

            attn_logits_softmax = torch.softmax(attn_logits, dim=-1)
            
            attn_logits = attn_logits.masked_fill(attn_logits == 0, float('-inf'))
            attn = F.softmax(
                attn_logits.transpose(1, 2).reshape(B, N_kv, self.num_heads * N_q)
            , dim=-1).view(B, N_kv, self.num_heads, N_q).transpose(1, 2)                # Shape: [batch_size, num_heads, num_inputs, num_slots].
            # attn_1 = attn*mask.float()
            attn = torch.nan_to_num(attn, nan=0.0)
            attn_vis = attn.sum(1)                                                      # Shape: [batch_size, num_inputs, num_slots].
            # attn_vis = attn.squeeze(1)  

            
            # Weighted mean.

            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            updates = torch.matmul(attn.transpose(-1, -2), v)                           # Shape: [batch_size, num_heads, num_slots, slot_size // num_heads].
            updates = updates.transpose(1, 2).reshape(B, N_q, -1)                       # Shape: [batch_size, num_slots, slot_size].
            
            slots = slots_prev + updates
            slots = slots + self.mlp(self.norm_mlp(slots))
        
        return slots, attn_vis, attn_logits, score,k

class SlotAttentionEncoder(nn.Module):
    
    def __init__(self, num_iterations, num_slots,
                 input_channels,slot_size, mlp_hidden_size, pos_channels, truncate='bi-level', init_method='embedding',num_heads = 1, drop_path = 0.0,):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_channels = input_channels
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.pos_channels = pos_channels
        self.init_method = init_method
        self.projector_x = linear(input_channels, slot_size, bias=False)
        self.project_slot = linear(slot_size, slot_size, bias=False)

        self.layer_norm = nn.LayerNorm(input_channels)
        self.mlp = nn.Sequential(
            linear(input_channels, input_channels, weight_init='kaiming'),
            nn.ReLU(),
            linear(input_channels, input_channels))
        
        assert init_method in ['shared_gaussian', 'embedding','mu_embedding']
        if init_method == 'shared_gaussian':
            # Parameters for Gaussian init (shared by all slots).
            self.slot_mu = nn.Parameter(torch.zeros(1, 1, slot_size))
            self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_size))
            nn.init.xavier_uniform_(self.slot_mu)
            nn.init.xavier_uniform_(self.slot_log_sigma)
        elif init_method == 'embedding':
            self.slots_init = nn.Embedding(num_slots, slot_size)
            nn.init.xavier_uniform_(self.slots_init.weight)
        elif init_method == 'mu_embedding':
            self.slots_mu = nn.Parameter(torch.randn(1,num_slots, slot_size))

        else:
            raise NotImplementedError
        
        self.slot_attention = SlotAttention(
            num_iterations,
            input_channels, slot_size, mlp_hidden_size, truncate, num_heads, drop_path=drop_path)
    
    def forward(self, x):
        # `image` has shape: [batch_size, img_channels, img_height, img_width].
        # `encoder_grid` has shape: [batch_size, pos_channels, enc_height, enc_width].
        B, N, _ = x.size()
        dtype = x.dtype
        device = x.device
        x = self.mlp(self.layer_norm(x))
        # `x` has shape: [batch_size, enc_height * enc_width, cnn_hidden_size].

        # Slot Attention module.
        if self.init_method == 'mu_embedding':
            init_slots = self.slots_mu.expand(B, -1, -1)
        else:
            init_slots = self.slots_initialization(B, dtype, device)

        

        slots, attn, attn_logits, score,imputs_mlp = self.slot_attention(x, init_slots)
        # `slots` has shape: [batch_size, num_slots, slot_size].
        # `attn` has shape: [batch_size, enc_height * enc_width, num_slots].
        
        return slots, attn, init_slots, attn_logits, score,imputs_mlp
    
    def slots_initialization(self, B, dtype, device):
        # The first frame, initialize slots.
        if self.init_method == 'shared_gaussian':
            slots_init = torch.empty((B, self.num_slots, self.slot_size), dtype=dtype, device=device).normal_()
            slots_init = self.slot_mu + torch.exp(self.slot_log_sigma) * slots_init
        elif self.init_method == 'embedding':

            slots_init = self.slots_init(torch.arange(0, self.num_slots, device=device)).unsqueeze(0).repeat(B, 1, 1)
        
        return slots_init
