''' Based on SPOT libraries:
https://github.com/gkakogeorgiou/spot.git
'''
from utils_spot import *
from slot_attn import SlotAttentionEncoder
from transformer_dec import TransformerDecoder
from mlp import MlpDecoder
import torch
import random
import math
import torch.nn as nn
import copy


import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.nn import CrossEntropyLoss
IGNORE_INDEX = -100


class FiLM(nn.Module):
    def __init__(self, feature_dim):
        super(FiLM, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, 1, feature_dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, feature_dim))

    def forward(self, x):
    
        return self.gamma * x + self.beta, self.gamma ,self.beta

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
    

class AttentionFusion(nn.Module):
    def __init__(self, slot_dim,dropout=0.1):
        super(AttentionFusion, self).__init__()
  
        self.query_fc = nn.Linear(slot_dim, slot_dim)
        self.key_fc = nn.Linear(slot_dim, slot_dim)
        self.value_fc = nn.Linear(slot_dim, slot_dim)
        self.slot_dim = slot_dim
        self.norm = nn.LayerNorm(slot_dim)
        self.fusion_fc = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, slot, fb_slot):
        B, N, D = slot.size()
        query = self.query_fc(slot) 
        key = self.key_fc(fb_slot)  
        value = self.value_fc(fb_slot)  
        
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(D)
        attention_weights = F.softmax(attention_scores, dim=-1)  

        attended_values = torch.matmul(attention_weights, value) 

        output_slot = slot + attended_values  # [batchsize, slot_num, slot_dim
        return output_slot



class SPOT(nn.Module):
    def __init__(self, encoder, encoder_s,args, second_encoder=None):
        super().__init__()

        self.which_encoder = args.which_encoder
        self.encoder = encoder
        self.second_encoder = second_encoder
        self.encoder_final_norm = args.encoder_final_norm
        
        for param_name, param in self.encoder.named_parameters():
            if ('blocks' in param_name):
                block_id = int(param_name.split('.')[1])
                if block_id >= args.finetune_blocks_after:
                    param.requires_grad = True  # update by gradient
                else:
                    param.requires_grad = False  # not update by gradient
            else:
                param.requires_grad = False  # not update by gradient
        

        if self.second_encoder is not None:
            for param in self.second_encoder.parameters():
                param.requires_grad = False  # not update by gradient

        # Estimate number of tokens for images of size args.image_size and
        # embedding size (d_model)
        with torch.no_grad():
            x = torch.rand(1, args.img_channels, args.image_size, args.image_size)
            x = self.forward_encoder(x, self.encoder)
            _, num_tokens, d_model = x.shape

        args.d_model = d_model

        self.num_slots = args.num_slots
        self.d_model = args.d_model

        self.slot_attn = SlotAttentionEncoder(
            args.num_iterations, args.num_slots,
            args.d_model, args.slot_size, args.mlp_hidden_size, args.pos_channels,
            args.truncate, args.init_method)
        
        
        self.slot_attn_s =copy.deepcopy(self.slot_attn)
        for p in self.slot_attn_s.parameters():
            p.requires_grad = False

        self.input_proj = nn.Sequential(
            linear(args.d_model, args.d_model, bias=False),
            nn.LayerNorm(args.d_model),
        )
        
        size = int(math.sqrt(num_tokens))
        standard_order = torch.arange(size**2) # This is the default "left_top"
        
        self.cappa = args.cappa
        self.train_permutations = args.train_permutations
        
        if self.train_permutations == 'standard':
            self.permutations = [standard_order]
            self.eval_permutations = 'standard'
        
        else:
            standard_order_2d = standard_order.reshape(size,size)
            
            perm_top_left = torch.tensor([standard_order_2d[row,col] for col in range(0, size, 1) for row in range(0, size, 1)])
            
            perm_top_right = torch.tensor([standard_order_2d[row,col] for col in range(size-1, -1, -1) for row in range(0, size, 1)])
            perm_right_top = torch.tensor([standard_order_2d[row,col] for row in range(0, size, 1) for col in range(size-1, -1, -1)])
            
            perm_bottom_right = torch.tensor([standard_order_2d[row,col] for col in range(size-1, -1, -1) for row in range(size-1, -1, -1)])
            perm_right_bottom = torch.tensor([standard_order_2d[row,col] for row in range(size-1, -1, -1) for col in range(size-1, -1, -1)])
            
            perm_bottom_left = torch.tensor([standard_order_2d[row,col] for col in range(0, size, 1) for row in range(size-1, -1, -1)])
            perm_left_bottom = torch.tensor([standard_order_2d[row,col] for row in range(size-1, -1, -1) for col in range(0, size, 1)])
            
            perm_spiral = spiral_pattern(standard_order_2d, how = 'top_right')
            perm_spiral = torch.tensor((perm_spiral[::-1]).copy())
    
            self.permutations = [standard_order, # left_top
                                 perm_top_left, 
                                 perm_top_right, 
                                 perm_right_top, 
                                 perm_bottom_right, 
                                 perm_right_bottom,
                                 perm_bottom_left,
                                 perm_left_bottom,
                                 perm_spiral
                                 ]
            self.eval_permutations = args.eval_permutations

        self.perm_ind = list(range(len(self.permutations)))

        self.bos_tokens = nn.Parameter(torch.zeros(len(self.permutations), 1, 1, args.d_model))
        torch.nn.init.normal_(self.bos_tokens, std=.02)
        
        self.dec_type = args.dec_type
        self.use_slot_proj = args.use_slot_proj
        
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
            self.remove_handle = self.dec._modules["blocks"][-1]._modules["encoder_decoder_attn"]._modules["attn_dropout"].register_forward_pre_hook(hook_fn_forward_attn)

        self.projector_fusion = nn.Linear(self.d_model*2, self.d_model)
  
        self.slot_fusion_module = AttentionFusion(slot_dim=args.slot_size)
        self.projector_slot_fusion = nn.Linear(args.slot_size,args.slot_size)
        self.projector_embed= nn.Linear(self.d_model,self.d_model)
        self.film = FiLM(feature_dim=768)
        self.film_t = FiLM(feature_dim=768)
        for p in self.film_t.parameters():
            p.requires_grad = False
        
        self.criterion = CrossEntropyLoss(ignore_index=IGNORE_INDEX)



    def forward_encoder(self, x, encoder):
        encoder.eval()

        if self.which_encoder in ['dinov2_vitb14', 'dinov2_vits14', 'dinov2_vitb14_reg', 'dinov2_vits14_reg']:
            x = encoder.prepare_tokens_with_masks(x, None)
        else:
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

            bos_token = self.bos_tokens[perm_id]
            bos_token = bos_token.expand(emb_target.shape[0], -1, -1)
            
            use_pos_emb = self.cappa > 0
            parallel_dec = self.cappa > 0 and ((self.cappa >= 1.0) or (self.training and random.random() < self.cappa))
            #print(f"Paralled Decoder (CAPPA) {parallel_dec}")
            # Input to the decoder
            if parallel_dec: # Use parallel decoder
                dec_input = self.mask_token.to(emb_target.dtype).expand(emb_target.shape[0], -1, -1)
            else: # Use autoregressive decoder
                dec_input = torch.cat((bos_token, emb_target[:,current_perm,:][:, :-1, :]), dim=1)
      
            if use_pos_emb:
                # Add position embedding if they exist.
                dec_input = dec_input + self.pos_embed.to(emb_target.dtype)

            # dec_input has the same shape as emb_target, which is [B, N, D]
            dec_input = self.input_proj(dec_input)
    
            # Apply the decoder
            dec_input_slots = self.slot_proj(slots) # shape: [B, num_slots, D]
            if self.dec_type=='transformer':
                dec_output = self.dec(dec_input, dec_input_slots, causal_mask=(not parallel_dec))
                # decoder_output shape [B, N, D]

                dec_slots_attns = self.dec_slots_attns[0]
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


   

    def one_hot(self,mask, num_classes):
        return F.one_hot(mask, num_classes).permute(0, 3, 1, 2).float()

    def iou_loss(self,pred, target, eps=1e-6):
        inter = torch.sum(pred * target, dim=(1,2))  # [C]
        union = torch.sum(pred + target, dim=(1,2)) - inter  # [C]
        iou = (inter + eps) / (union + eps)
        return 1 - iou.mean() 
    
    def dice_loss(self, pred, target, eps=1e-6):
        # pred, target: [C, H, W]
        inter = (pred * target).sum(dim=(1,2))
        union = pred.sum(dim=(1,2)) + target.sum(dim=(1,2)) + eps
        dice = 2 * inter / union
        return 1 - dice.mean()


    def hungarian_ce_loss(self, mask1, mask2,logits):
        B, C,_, H, W = mask1.shape
        mask1 = mask1.squeeze(2)
        mask2 = mask2.squeeze(2)
        total_loss = 0.0

        for b in range(B):
            c1 = mask1[b]  # [C, H, W]
            c2 = mask2[b]  # [C, H, W]
            logits_b = logits[b] #[C,H,W]


            c1_flat = c1.view(C, -1)
            c2_flat = c2.view(C, -1)


            inter = torch.matmul(c1_flat, c2_flat.T)  # [C, C]
            area1 = c1_flat.sum(dim=1, keepdim=True)  # [C, 1]
            area2 = c2_flat.sum(dim=1, keepdim=True).T  # [1, C]
            union = area1 + area2 - inter + 1e-6
            cost = 1 - inter / union  # [C, C]
            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())


            matched_idx = torch.tensor(col_ind, dtype=torch.long, device=c2.device)
            aligned_c2 = c2[matched_idx] 

            target_label = c1.argmax(0)  # [H, W]
            pred_prob = aligned_c2.unsqueeze(0)  # [1, C, H, W]
            loss = F.cross_entropy(pred_prob, target_label.unsqueeze(0).long())
            total_loss += loss

        return total_loss / B



    @torch.no_grad()
    def update_slot_attn_s(self, ema=False, decay=0.99):
        for param, ema_param in zip(self.slot_attn.parameters(), self.slot_attn_s.parameters()):
            if ema:
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
            else:
                ema_param.data.copy_(param.data)
    def update_dec_s(self, ema=False, decay=0.99):
        for param, ema_param in zip(self.dec.parameters(), self.dec_s.parameters()):
            if torch.is_floating_point(ema_param.data):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
            else:
                ema_param.data.copy_(param.data)

    def compare_param(self):
        for (n1, p1), (n2, p2) in zip(self.slot_attn.named_parameters(), self.slot_attn_s.named_parameters()):
                if not torch.allclose(p1, p2, atol=1e-5):
                    print(f"Different param: {n1} vs {n2}")
                    break
                else:
                    print("All parameters are equal.")


    def att_matching(self,attention_1, attention_2):

        batch_size, slots, height, width = attention_1.shape
        
        mask_1 = torch.nn.functional.one_hot(attention_1.argmax(1).reshape(batch_size,-1), num_classes=slots).to(torch.float32).permute(0,2,1)
        mask_2 = torch.nn.functional.one_hot(attention_2.argmax(1).reshape(batch_size,-1), num_classes=slots).to(torch.float32).permute(0,2,1)
        
        # assumes shape: batch_size, set_size, channels
        is_padding = (mask_2 == 0).all(-1)

        # discretized 2d mask for hungarian matching
        pred_mask_1_id = torch.argmax(mask_1, -2)
        pred_mask_1_disc = rearrange(
            F.one_hot(pred_mask_1_id, mask_1.size(1)).to(torch.float32), "b c n -> b n c"
        )

        # treat as if no padding in mask_2
        pIoU = pairwise_IoU_efficient(pred_mask_1_disc.float(), mask_2.float())
        pIoU_inv = 1 - pIoU
        pIoU_inv[is_padding] = 1e3
        pIoU_inv_ = pIoU_inv.detach().cpu().numpy()

        # hungarian matching
        indices = np.array([linear_sum_assignment(p)[1] for p in pIoU_inv_])
        #attention_2_permuted = torch.stack([x[indices[n]] for n, x in enumerate(attention_2)],dim=0)

        pIoU = pIoU.detach().cpu().numpy()
        matching_scores = np.array([[pIoU[b][i,j] for i,j in enumerate(indices[b])] for b in range(batch_size)])
        return indices, matching_scores
    
    def init_slot_attn_s(self):
        self.slot_attn_s.load_state_dict(self.slot_attn.state_dict())
        for param in self.slot_attn_s.parameters():
            param.requires_grad = False  # 冻结梯度
                
    def forward_ours_stage2(self, image, emb_input_teacher,slots_teacher):
            """
            image: batch_size x img_channels x H x W
            """
            B, _, H, W = image.size()
            emb_input = self.forward_encoder(image, self.encoder)
            with torch.no_grad():
                if self.second_encoder is not None:
                    emb_target = self.forward_encoder(image, self.second_encoder)
                    # emb_target = emb_input.clone().detach()

                else:
                    emb_target = emb_input.clone().detach()
            # emb_target shape: B, N, D
            slots, slots_attns, init_slots, attn_logits,_ = self.slot_attn(emb_input)
            slots_fusion = self.slot_fusion_module(slots,slots_teacher)


            dec_recon, dec_slots_attns  = self.forward_decoder(slots_fusion, emb_target)
            # dec_recon, dec_slots_attns  = self.forward_decoder(slots, emb_input_fusion)

            # MSE loss
            H_enc, W_enc = int(math.sqrt(emb_target.shape[1])), int(math.sqrt(emb_target.shape[1]))
            loss_mse = ((emb_target - dec_recon) ** 2).sum()/(B*H_enc*W_enc*self.d_model)

            # Second branch
            emb_input_s = self.forward_encoder(image, self.encoder)
            emb_input_s,gamma,beta = self.film(emb_input_s)
            emb_target_s = emb_input_s.clone().detach()

            slots_s, slots_attn_s, _,logits, slot_attns_s_22 = self.slot_attn_s(emb_input_s)


            dec_slots_attns = dec_slots_attns.transpose(-1, -2).reshape(B, self.num_slots, H_enc, W_enc)
            slots_attn_s = slots_attn_s.transpose(-1, -2).reshape(B, self.num_slots, H_enc, W_enc)
            logits = logits.transpose(-1, -2).reshape(B, self.num_slots, H_enc, W_enc)

            logits = F.interpolate(logits, size=224,mode='bilinear')


            # mask生成
            dec_attns = F.interpolate(dec_slots_attns, size=224,mode='bilinear')
            dec_attns = dec_attns.unsqueeze(2) # shape [B, num_slots, 1, H, W]
            pred_dec_mask = F.gumbel_softmax(dec_attns, tau=1.0, hard=True, dim=1)


            slot_attns_s = F.interpolate(slots_attn_s, size=224,mode='bilinear')
            slot_attns_s = slot_attns_s.unsqueeze(2) # shape [B, num_slots, 1, H, W]
            pred_slot_mask_s = F.gumbel_softmax(slot_attns_s, tau=1.0, hard=True, dim=1)

            loss_map = self.hungarian_ce_loss(pred_dec_mask,pred_slot_mask_s,logits)
            return loss_mse , loss_map ,  dec_slots_attns, slots_fusion, dec_recon


    def forward_ours_eval(self, image, emb_input_teacher,slots_teacher):
            """
            image: batch_size x img_channels x H x W
            """

            B, _, H, W = image.size()
            emb_input = self.forward_encoder(image, self.encoder)
            with torch.no_grad():
                if self.second_encoder is not None:
                    emb_target = self.forward_encoder(image, self.second_encoder)
                    # emb_target = emb_input.clone().detach()

                else:
                    emb_target = emb_input.clone().detach()
            # emb_target shape: B, N, D

            # Second branch
            emb_input_s,gamma,beta = self.film(emb_input)

            slots_s, _, _,_ ,_= self.slot_attn(emb_input_s)
            slots_fusion_s = self.slot_fusion_module(slots_s,slots_teacher)

            dec_recon, dec_slots_attns  = self.forward_decoder(slots_fusion_s, emb_target)

            # Mean-Square-Error loss
            H_enc, W_enc = int(math.sqrt(emb_target.shape[1])), int(math.sqrt(emb_target.shape[1]))
            loss_mse = ((emb_target - dec_recon) ** 2).sum()/(B*H_enc*W_enc*self.d_model)

            # Reshape the slot and decoder-slot attentions.
            dec_slots_attns = dec_slots_attns.transpose(-1, -2).reshape(B, self.num_slots, H_enc, W_enc)

            loss = loss_mse   
            return loss, dec_slots_attns, slots_fusion_s, dec_recon,gamma,beta