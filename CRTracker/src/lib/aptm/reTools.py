import torch
import torch.nn.functional as F


@torch.no_grad()
def inference_attr(model, text, images, tokenizer, device, config):
    model.eval()

    # Text Processing
    text_input = tokenizer(text, padding='longest', max_length=config['max_tokens'],
                           return_tensors="pt").to(device)
    text_embeds = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
    text_atts = text_input.attention_mask 

    images = torch.stack(images) # Multidimensional list -> torch
    images = images.to(device)
    image_embeds, _ = model.get_vision_embeds(images)

    score_matrix_i2t = torch.full((len(images), len(text)), -1000.0).to(device)

    for i, image_embed in enumerate(image_embeds): 
        encoder_output = image_embed.repeat(len(text), 1, 1) 
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device) 
        output = model.get_cross_embeds(encoder_output, encoder_att, text_embeds=text_embeds,
                                        text_atts=text_atts)[:, 0, :] 
        score = model.itm_head(output)[:, 1]
        score_matrix_i2t[i] = score

    score_matrix_i2t = score_matrix_i2t.t()

    return score_matrix_i2t.cpu().numpy() # torch.Size([1, len(images)])


@torch.no_grad()
def inference_text(model, text, images, CNN_image_features, CNN_image_alpha, tokenizer, device, config):
    model.eval()

    text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                            return_tensors="pt").to(device)
    text_embeds = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
    text_feats = model.text_proj(text_embeds[:, 0, :])
    text_feats = F.normalize(text_feats, dim=-1)

    # Multidimensional list -> torch
    images = torch.stack(images)
    images = images.to(device)
    image_embeds, _ = model.get_vision_embeds(images)
    image_feats = model.vision_proj(image_embeds[:, 0, :])
    image_feats = F.normalize(image_feats, dim=-1)

    # Fusion:
    # The coefficient before CNN_image_features is CNN_image_alpha
    # normalized features
    norm_CNN_image_features = CNN_image_features / CNN_image_features.norm(dim=1, keepdim=True)
    
    concat_image_feats = image_feats + CNN_image_alpha * norm_CNN_image_features

    score_sim_t2i = concat_image_feats @ text_feats.t()
    score_sim_t2i = score_sim_t2i.t()

    return score_sim_t2i.cpu().numpy() # torch.Size([1, len(images)])


# @torch.no_grad()
def train_text(model, texts, images, CNN_image_features, CNN_image_alpha, tokenizer, device, config):
    # model.eval()

    text_input = tokenizer(texts, padding='max_length', truncation=True, max_length=config['max_tokens'],
                            return_tensors="pt").to(device)
    text_embeds = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
    text_feats = model.text_proj(text_embeds[:, 0, :])
    text_feats = F.normalize(text_feats, dim=-1)

    # Multidimensional list -> torch
    images = torch.stack(images)
    images = images.to(device)
    image_embeds, _ = model.get_vision_embeds(images)
    image_feats = model.vision_proj(image_embeds[:, 0, :])
    image_feats = F.normalize(image_feats, dim=-1)

    # Fusion:
    # The coefficient before CNN_image_features is CNN_image_alpha
    # normalized features
    norm_CNN_image_features = CNN_image_features / CNN_image_features.norm(dim=1, keepdim=True)
    
    concat_image_feats = image_feats + CNN_image_alpha * norm_CNN_image_features

    score_sim_t2i = concat_image_feats @ text_feats.t()
    # score_sim_t2i = score_sim_t2i.t()

    # return score_sim_t2i.cpu().numpy() # torch.Size([len(texts), len(images)])
    return score_sim_t2i # torch.Size([len(texts), len(images)])
