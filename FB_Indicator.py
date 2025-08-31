from utils_spot import *
from slot_attn_feature import SlotAttentionEncoder
from transformer import TransformerDecoder
from mlp import MlpDecoder
import torch
import random
import math
# from data.transforms import CustomDataAugmentation
import torchvision


class DINOHead(nn.Module):
    def __init__(self, in_dim, use_bn=True, nlayers=3, hidden_dim=4096, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B,N,D = x.shape
        x = x.reshape(B*N, D)
        x = self.mlp(x)
        x = x.reshape(B, N, D)
        return x

class Indicator(nn.Module):
    def __init__(self, encoder, args, second_encoder=None):
        super().__init__()

        self.which_encoder = args.which_encoder
        self.encoder_s = encoder
        self.encoder_t = encoder

        self.second_encoder = second_encoder
        self.encoder_final_norm = args.encoder_final_norm
        # self.student_temp = args.student_temp
        
        for param_name, param in self.encoder_s.named_parameters():
            if ('blocks' in param_name):
                block_id = int(param_name.split('.')[1])
                if block_id >= args.finetune_blocks_after:
                    param.requires_grad = True  # update by gradient
                else:
                    param.requires_grad = False  # not update by gradient
            else:
                param.requires_grad = False # not update by gradient
        
        for param_s, param_t in zip(self.encoder_s.parameters(), self.encoder_t.parameters()):
            param_t.data.copy_(param_s.data)  # initialize
            param_t.requires_grad = False  # not update by gradient

            
        if self.second_encoder is not None:
            for param in self.second_encoder.parameters():
                param.requires_grad = False  # not update by gradient

        # Estimate number of tokens for images of size args.image_size and
        # embedding size (d_model)
        with torch.no_grad():
            x = torch.rand(1, args.img_channels, args.image_size, args.image_size)
            x = self.forward_encoder(x, self.encoder_s)
            _, num_tokens, d_model = x.shape

        args.d_model = d_model

        self.num_slots = args.num_slots
        self.d_model = args.d_model

        self.slot_attn_s = SlotAttentionEncoder(
            args.num_iterations, args.num_slots,
            args.d_model, args.slot_size, args.mlp_hidden_size, args.pos_channels,
            args.truncate, args.init_method)
        
        self.slot_attn_t = SlotAttentionEncoder(
            args.num_iterations, args.num_slots,
            args.d_model, args.slot_size, args.mlp_hidden_size, args.pos_channels,
            args.truncate, args.init_method)
        


        self.input_proj = nn.Sequential(
            linear(args.d_model, args.d_model, bias=False),
            nn.LayerNorm(args.d_model),
        )

        for param_s, param_t in zip(self.slot_attn_s.parameters(), self.slot_attn_t.parameters()):
            param_t.data.copy_(param_s.data)  # initialize
            param_t.requires_grad = False  # not update by gradient
        
        size = int(math.sqrt(num_tokens))
        standard_order = torch.arange(size**2) # This is the default "left_top"
        
        self.cappa = args.cappa
        self.train_permutations = args.train_permutations
        self.eval_permutations = args.eval_permutations
        self.permutations = [standard_order]

        self.perm_ind = list(range(len(self.permutations)))

        self.bos_tokens = nn.Parameter(torch.zeros(len(self.permutations), 1, 1, args.d_model))
        torch.nn.init.normal_(self.bos_tokens, std=.02)
        
        self.dec_type = args.dec_type
        self.use_slot_proj = args.use_slot_proj


        self.group_loss_weight = args.group_loss_weight
        self.ctr_loss_weight = args.ctr_loss_weight
        self.differ_loss_weight = args.differ_loss_weight
        self.center_momentum = args.center_momentum
        self.register_buffer("center", torch.zeros(1, self.num_slots))
        
        if self.dec_type=='mlp' and not self.use_slot_proj:
            self.slot_proj = nn.Identity()
            self.dec_input_dim = args.slot_size
        else:
            self.slot_proj = nn.Sequential(
                linear(args.slot_size, args.d_model, bias=False),
                nn.LayerNorm(args.d_model),
            )
            self.dec_input_dim = args.d_model
        
        if self.dec_type=='transformer':
            self.dec = TransformerDecoder(
                args.num_dec_blocks, args.max_tokens, args.d_model, args.num_heads, args.dropout, args.num_cross_heads)
            if self.cappa > 0:
                assert (self.train_permutations == 'standard') and (self.eval_permutations == 'standard')   
                self.mask_token = nn.Parameter(torch.zeros(1, 1, args.d_model))
                self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, args.d_model))
                torch.nn.init.normal_(self.pos_embed, std=.02)
                torch.nn.init.normal_(self.mask_token, std=.02)
                  
        elif self.dec_type=='mlp':
            self.dec = MlpDecoder(self.dec_input_dim, args.d_model, args.max_tokens, args.mlp_dec_hidden)

            assert (self.train_permutations == 'standard') and (self.eval_permutations == 'standard')  
        else:
            raise

        if self.dec_type=='transformer':
            # Register hook for capturing the cross-attention (of the query patch
            # tokens over the key/value slot tokens) from the last decoder
            # transformer block of the decoder.
            self.dec_slots_attns = []
            def hook_fn_forward_attn(module, input):
                self.dec_slots_attns.append(input[0])
            self.remove_handle_s = self.dec._modules["blocks"][-1]._modules["encoder_decoder_attn"]._modules["attn_dropout"].register_forward_pre_hook(hook_fn_forward_attn)
            # self.remove_handle_t = self.dec_t._modules["blocks"][-1]._modules["encoder_decoder_attn"]._modules["attn_dropout"].register_forward_pre_hook(hook_fn_forward_attn)

        self.kernel_size = args.kernel_size           
        self.top_k = args.top_k
        
        self.mlp_hidden_size = args.mlp_hidden_size
        self.projector_s = DINOHead(self.d_model, hidden_dim=self.mlp_hidden_size, bottleneck_dim=self.d_model)
        self.projector_t = DINOHead(self.d_model, hidden_dim=self.mlp_hidden_size, bottleneck_dim=self.d_model)
        self.teacher_momentum = args.teacher_momentum

        for param_s, param_t in zip(self.projector_s.parameters(), self.projector_t.parameters()):
            param_t.data.copy_(param_s.data)  # initialize
            param_t.requires_grad = False  # not update by gradient

        self.K = int(args.num_instances * 1. / 1 / args.batch_size * args.epochs)
        self.k = int(args.num_instances * 1. / 1 / args.batch_size * (args.start_epoch - 1))
        self.teacher_temp = args.teacher_temp
        self.student_temp = args.student_temp

        self.mlp = nn.Sequential(
            linear(args.d_model, args.slot_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(args.slot_size, args.slot_size))
        
        self.layer_norm = nn.LayerNorm(args.d_model)

    def forward_encoder(self, x, encoder):
        encoder.train()

        x = encoder.prepare_tokens(x)

        for blk in encoder.blocks:
            x = blk(x)
        if self.encoder_final_norm: # The DINOSAUR paper does not use the final norm layer according to the supplementary material.
            x = encoder.norm(x)
        
        offset = 1
        if self.which_encoder in ['dinov2_vitb14_reg', 'dinov2_vits14_reg']:
            offset += encoder.num_register_tokens
        elif self.which_encoder in ['simpool_vits16']:
            offset += -1
        x = x[:, offset :] # remove the [CLS] and (if they exist) registers tokens 

        return x
    
    def forward_decoder(self, slots, emb_target):
        # Prepate the input tokens for the decoder transformer:
        # (1) insert a learnable beggining-of-sequence ([BOS]) token at the beggining of each target embedding sequence.
        # (2) remove the last token of the target embedding sequence
        # (3) no need to add positional embeddings since positional information already exists at the DINO's outptu.
        

        if self.training:
            if self.train_permutations == 'standard':
                which_permutations = [0] # USE [0] FOR THE STANDARD ORDER
            elif self.train_permutations == 'random':
                which_permutations = [random.choice(self.perm_ind)]
            elif self.train_permutations == 'all':
                which_permutations = self.perm_ind
            else:
                raise
        else:
            if self.eval_permutations == 'standard':
                which_permutations = [0] # USE [0] FOR THE STANDARD ORDER
            elif self.eval_permutations == 'random':
                which_permutations = [random.choice(self.perm_ind)]
            elif self.eval_permutations == 'all':
                which_permutations = self.perm_ind
            else:
                raise
        
        
        all_dec_slots_attns = []
        all_dec_output = []
        
        for perm_id in which_permutations:
            current_perm = self.permutations[perm_id]

            bos_token = self.bos_tokens[perm_id]  # [1,1,768]
            bos_token = bos_token.expand(emb_target.shape[0], -1, -1) # [32,1,768]
            
            use_pos_emb = self.cappa > 0
            parallel_dec = self.cappa > 0 and ((self.cappa >= 1.0) or (self.training and random.random() < self.cappa))
            #print(f"Paralled Decoder (CAPPA) {parallel_dec}")
            # Input to the decoder
            if parallel_dec: # Use parallel decoder
                dec_input = self.mask_token.to(emb_target.dtype).expand(emb_target.shape[0], -1, -1)
            else: # Use autoregressive decoder
                # dec_input [B,N,D]
                dec_input = torch.cat((bos_token, emb_target[:,current_perm,:][:, :-1, :]), dim=1)
      
            if use_pos_emb:
                # Add position embedding if they exist.
                dec_input = dec_input + self.pos_embed.to(emb_target.dtype)

            # dec_input has the same shape as emb_target, which is [B, N, D]
            dec_input = self.input_proj(dec_input)
    
            # Apply the decoder
            dec_input_slots = self.slot_proj(slots) # shape: [B, num_slots, 768]  slots [B, num_slots, 256]
            if self.dec_type=='transformer':
                dec_output = self.dec(dec_input, dec_input_slots, causal_mask=(not parallel_dec))
                # dec_output = dec(dec_input, dec_input_slots, causal_mask=(not parallel_dec))
                # decoder_output shape [B, N, D], reconstruction image

                dec_slots_attns = self.dec_slots_attns[0] # [32,6,196,7]
                self.dec_slots_attns = []

                # sum over the heads and 
                dec_slots_attns = dec_slots_attns.sum(dim=1) # [B, N, num_slots]
                # dec_slots_attns shape [B, num_heads, N, num_slots]
                # L1-normalize over the slots so as to sum to 1.
                dec_slots_attns = dec_slots_attns / dec_slots_attns.sum(dim=2, keepdim=True)

                inv_current_perm = torch.argsort(current_perm)
                dec_slots_attns = dec_slots_attns[:,inv_current_perm,:]
                dec_output = dec_output[:,inv_current_perm,:]

            elif self.dec_type=='mlp':
                dec_output, dec_slots_attns = self.dec(dec_input_slots)
                dec_slots_attns = dec_slots_attns.transpose(1,2)

            else:
                raise
            
            all_dec_slots_attns.append(dec_slots_attns)
            all_dec_output.append(dec_output)

        mean_dec_slots_attns = torch.stack(all_dec_slots_attns).mean(0)
        mean_dec_output = torch.stack(all_dec_output).mean(0)

        return mean_dec_output, mean_dec_slots_attns

    def get_embeddings_n_slots(self, image):
        """
        image: batch_size x img_channels x H x W
        """

        B, _, H, W = image.size()
        with torch.no_grad():
            emb_target = self.forward_encoder(image, self.encoder_s)
        # emb_target shape: B, N, D

        # Apply the slot attention
        slots, slots_attns, _ = self.slot_attn(emb_target)
        return emb_target, slots, slots_attns
    
    def invaug(self, x, coords, flags):
        N, C, H, W = x.shape

        batch_idxs = torch.arange(N, device=coords.device).view(N, 1)
        coords_rescaled = coords.clone()
        coords_rescaled[:, 0] = coords_rescaled[:, 0] * W  # x1
        coords_rescaled[:, 2] = coords_rescaled[:, 2] * W  # x2
        coords_rescaled[:, 1] = coords_rescaled[:, 1] * H  # y1
        coords_rescaled[:, 3] = coords_rescaled[:, 3] * H  # y2
        coords_with_idxs = torch.cat([batch_idxs, coords_rescaled], dim=1)
        
        x_aligned = torchvision.ops.roi_align(x, coords_with_idxs, (H, W), aligned=True)
        x_flipped = torch.stack([feat.flip(-1) if flag else feat for feat, flag in zip(x_aligned, flags)])
        return x_flipped

    def self_distill(self, q, k):
        q = F.log_softmax(q / self.student_temp, dim=-1)
        k = F.softmax((k - self.center) / self.teacher_temp, dim=-1)
        # k = F.softmax((k) / self.teacher_temp, dim=-1)
        return torch.sum(-k * q, dim=-1).mean()
    # k,q shape[B,slot_num,D],score[B,slot_num,H,W]
    
    def ctr_loss_filtered(self, q, k, score_q, score_k, tau=0.2):
        q = q.flatten(0, 1)
        k = F.normalize(k.flatten(0, 1), dim=1)

        mask_q = (torch.zeros_like(score_q).scatter_(1, score_q.argmax(1, keepdim=True), 1).sum(-1).sum(-1) > 0).long().detach()
        mask_k = (torch.zeros_like(score_k).scatter_(1, score_k.argmax(1, keepdim=True), 1).sum(-1).sum(-1) > 0).long().detach()
        mask_intersection = (mask_q * mask_k).view(-1)
        idxs_q = mask_intersection.nonzero().squeeze(-1)

        mask_k = mask_k.view(-1)
        idxs_k = mask_k.nonzero().squeeze(-1)

        N = k.shape[0]
        logits = torch.einsum('nc,mc->nm', [F.normalize(q[idxs_q], dim=1), k[idxs_k]]) / tau

        labels = mask_k.cumsum(0)[idxs_q] - 1

    

        return F.cross_entropy(logits, labels) * (2 * tau)

    def ctr_loss_fg(self, q, k, tau=0.2):

        device = q.device
        q = F.normalize(q.flatten(0, 1), dim=1)
        k = F.normalize(k.flatten(0, 1), dim=1)
        N = k.shape[0]

        idxs_bg = [i for i in range(N) if i % 2 != 0]
        idxs_fg = [i for i in range(N) if i % 2 == 0]


        logits_BG_pos = torch.einsum('nc,mc->nm', [F.normalize(q[idxs_bg], dim=1), F.normalize(k[idxs_bg], dim=1)]) 
        logits_BG_neg = torch.einsum('nc,mc->nm', [F.normalize(q[idxs_bg], dim=1), F.normalize(k[idxs_fg], dim=1)]) 

        target_BG_pos = torch.ones_like(logits_BG_pos)
        target_BG_neg = torch.zeros_like(logits_BG_neg)

        ctr_pos = F.mse_loss(logits_BG_pos,target_BG_pos) 
        ctr_neg = F.mse_loss(logits_BG_neg,target_BG_neg)

        return ctr_neg + ctr_pos

    

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
    
        batch_center = batch_center / len(teacher_output)

        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    @torch.no_grad()
    def _momentum_update_teacher_encoder(self):
        """
        Momentum update of the key encoder
        """
        momentum = 1. - (1. - self.teacher_momentum) * (math.cos(math.pi * self.k / self.K) + 1.) * 0.5
        self.k += 1
        for param_s, param_t in zip(self.encoder_s.parameters(), self.encoder_t.parameters()):
            param_t.data = param_t.data * momentum + param_s.data * (1. - momentum)
        for param_s, param_t in zip(self.projector_s.parameters(), self.projector_t.parameters()):
            param_t.data = param_t.data * momentum + param_s.data * (1. - momentum)
        for param_s, param_t in zip(self.slot_attn_s.parameters(), self.slot_attn_t.parameters()):
            param_t.data = param_t.data * momentum + param_s.data * (1. - momentum) 

    def generate_prototype(self,slots_attns,emb_target):
        max_values, max_indices = torch.max(slots_attns, dim=2)

        mask_fg = (max_indices == 0).unsqueeze(1).float()  
        mask_bg = (max_indices == 1).unsqueeze(1).float() 

        foreground_features = emb_target * mask_fg.transpose(1, 2)
        mean_foreground_features = foreground_features.sum(dim=1).unsqueeze(1) / mask_fg.sum(dim=-1, keepdim=True).clamp(min=1)

        background_features = emb_target * mask_bg.transpose(1, 2)
        mean_background_features = background_features.sum(dim=1).unsqueeze(1) / mask_fg.sum(dim=-1, keepdim=True).clamp(min=1)

        prototype_fg = self.mlp(self.layer_norm(mean_foreground_features))
        prototype_bg = self.mlp(self.layer_norm(mean_background_features))

        return  torch.stack((prototype_fg, prototype_bg), dim=1).squeeze(2)
    
    def compute_differ_loss(attn_logits, num_slots):
        attn = attn_logits.squeeze(1).softmax(dim=-1) + 1e-8   
        attn = F.normalize(attn, dim=-2)                       
    
        sim = torch.einsum('bcm,bcn->bmn', attn, attn)         # [B, num_slots, num_slots]

        mask = torch.ones_like(sim)
        mask[:, torch.arange(num_slots), torch.arange(num_slots)] = 0

        return torch.sum(sim * mask) / torch.sum(mask)


    def forward(self, input):
        """
        image: batch_size x img_channels x H x W
        """
        crops, coords, flags = input
        
        B, _, H, W = crops[0].shape
        # emb_input shape [B,H_ENC*W_ENC,dim],[256,196,768]
        emb_input_x1, emb_input_x2 = self.projector_s(self.forward_encoder(crops[0], self.encoder_s)), self.projector_s(self.forward_encoder(crops[1], self.encoder_s))
        with torch.no_grad():
            self._momentum_update_teacher_encoder()
            emb_input_y1, emb_input_y2 = self.projector_t(self.forward_encoder(crops[0], self.encoder_t)), self.projector_t(self.forward_encoder(crops[1], self.encoder_t))

        # input_x1, input_x2=self.projector(emb_input_x1),self.projector(emb_input_x1)
        with torch.no_grad():
            emb_target_x1,emb_target_x2 = emb_input_x1.clone().detach(), emb_input_x2.clone().detach()
            emb_target_y1,emb_target_y2 = emb_input_y1.clone().detach(), emb_input_y2.clone().detach()
        # emb_target shape: B, N, D

        # Apply the slot attention
        (slots_x1, slots_attns_x1, init_slot_x1, attn_logits_x1,score_x1,_), (slots_x2, slots_attns_x2, init_slot_x2, attn_logits_x2,score_x2,_) = self.slot_attn_s(emb_input_x1),self.slot_attn_s(emb_input_x2)
        
        prototype_x1,prototype_x2 = self.generate_prototype(slots_attns_x1,emb_input_x1),self.generate_prototype(slots_attns_x2,emb_input_x2)

        x1_aligned, x2_aligned = self.invaug(score_x1, coords[0], flags[0]), self.invaug(score_x2, coords[1], flags[1])
        with torch.no_grad():    
            (slots_y1, slots_attns_y1, init_slot_y1, attn_logits_y1,score_y1,_), (slots_y2, slots_attns_y2, init_slot_y2, attn_logits_y2,score_y2,_) = self.slot_attn_t(emb_input_y1),self.slot_attn_t(emb_input_y2)
            prototype_y1,prototype_y2 = self.generate_prototype(slots_attns_y1,emb_input_y1),self.generate_prototype(slots_attns_y2,emb_input_y2)
            y1_aligned, y2_aligned = self.invaug(score_y1, coords[0], flags[0]), self.invaug(score_y2, coords[1], flags[1])

        group_loss = self.group_loss_weight * self.self_distill(x1_aligned.permute(0, 2, 3, 1).flatten(0, 2), y2_aligned.permute(0, 2, 3, 1).flatten(0, 2)) \
             + self.group_loss_weight * self.self_distill(x2_aligned.permute(0, 2, 3, 1).flatten(0, 2), y1_aligned.permute(0, 2, 3, 1).flatten(0, 2))

        self.update_center(torch.cat([score_y1, score_y2]).permute(0, 2, 3, 1).flatten(0, 2))

        ctr_loss =  self.ctr_loss_weight * self.ctr_loss_fg(prototype_x1,prototype_y2) \
                + self.ctr_loss_weight * self.ctr_loss_fg(prototype_x2,prototype_y1)


        differ_loss_x1 = self.compute_differ_loss(attn_logits_x1, self.num_slots)
        differ_loss_x2 = self.compute_differ_loss(attn_logits_x2, self.num_slots)
        differ_loss_y1 = self.compute_differ_loss(attn_logits_y1, self.num_slots)
        differ_loss_y2 = self.compute_differ_loss(attn_logits_y2, self.num_slots)

        differ_loss_x = differ_loss_x1 + differ_loss_x2
        differ_loss_y = differ_loss_y1 + differ_loss_y2

        differ_loss =self.differ_loss_weight*(differ_loss_x + differ_loss_y) 
        
        loss = group_loss + ctr_loss + differ_loss

        H_enc, W_enc = int(math.sqrt(emb_target_y1.shape[1])), int(math.sqrt(emb_target_y1.shape[1]))


        # Reshape the slot and decoder-slot attentions.
        slots_attns_y1 = slots_attns_y1.transpose(-1, -2).reshape(B, self.num_slots, H_enc, W_enc)

        return loss, group_loss, ctr_loss,differ_loss,slots_attns_y1,



    def forward_eval(self, image):
        """
        image: batch_size x img_channels x H x W
        """
        B, _, H, W = image.size()
        emb_input = self.projector_t(self.forward_encoder(image, self.encoder_t))       

        with torch.no_grad():
            if self.second_encoder is not None:
                emb_target = self.forward_encoder(image, self.second_encoder)
            else:
                emb_target = emb_input.clone().detach()
        # emb_target shape: B, N, D

        # Apply the slot attention
        slots, slots_attns, init_slots, attn_logits,_,emb_input_mlp = self.slot_attn_t(emb_input)
        # attn_logits = attn_logits.squeeze()
        # slots shape: [B, num_slots, Ds]
        # slots_attns shape: [B, N, num_slots]
        prototype_mean = self.generate_prototype(slots_attns,emb_input)


        # Mean-Square-Error loss
        H_enc, W_enc = int(math.sqrt(emb_target.shape[1])), int(math.sqrt(emb_target.shape[1]))


        # Reshape the slot and decoder-slot attentions.
        slots_attns = slots_attns.transpose(-1, -2).reshape(B, self.num_slots, H_enc, W_enc)

        return emb_input, slots_attns,  slots,  attn_logits,emb_input_mlp,prototype_mean