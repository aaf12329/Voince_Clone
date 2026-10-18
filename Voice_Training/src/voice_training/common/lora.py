"""Teacher-forced codec distributions and small, mergeable LoRA adapters.

Uses the installed Qwen model's components, with one explicit next-token shift.
Language is passed from the role configuration; default Chinese preserves prior behavior.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        """Initialize the instance state and resources owned by this component."""
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, value):
        """Apply the frozen projection plus the scaled trainable low-rank update."""
        return self.base(value) + (F.linear(F.linear(value.float(), self.a), self.b) * self.scale).to(value.dtype)

    @property
    def weight(self):
        """Expose the wrapped projection weight to the Qwen model components."""
        return self.base.weight

    def merge(self):
        """Fold the low-rank update into the base projection and return that layer."""
        with torch.no_grad():
            self.base.weight.add_((self.b @ self.a * self.scale).to(self.base.weight.dtype))
        return self.base


def attach_lora(model, rank, alpha):
    """Freeze the backbone and replace selected talker projections with adapters."""
    model.requires_grad_(False)
    targets = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    attached = []
    for name, module in list(model.talker.named_modules()):
        if isinstance(module, nn.Linear) and name.rsplit('.', 1)[-1] in targets:
            parent_name, child = name.rsplit('.', 1)
            parent = model.talker.get_submodule(parent_name)
            setattr(parent, child, LoRALinear(module, rank, alpha))
            attached.append('talker.' + name)
    if not attached:
        raise RuntimeError('没有找到可训练的 LoRA 层。')
    return attached


def adapter_state(model):
    """Copy trainable adapter tensors to CPU for checkpoint persistence."""
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def load_adapter(model, values):
    """Validate checkpoint adapter names and restore matching trainable tensors."""
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(values) != expected:
        raise ValueError('检查点的适配器层与当前模型不一致。')
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in values:
                parameter.copy_(values[name].to(parameter.device))


def merge_lora(model):
    """Merge all talker adapters into their original linear layers."""
    for name, module in list(model.talker.named_modules()):
        if isinstance(module, LoRALinear):
            parent_name, child = name.rsplit('.', 1)
            setattr(model.talker.get_submodule(parent_name), child, module.merge())


def text_ids(processor, text):
    """Tokenize the assistant wrapper and return aligned role and content IDs."""
    wrapped = f'<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n'
    ids = processor(text=wrapped, return_tensors='pt')['input_ids'][0]
    if ids.numel() <= 8:
        raise ValueError('文本编码为空。')
    return ids[:3], ids[3:-5]


def build_inputs(model, role_ids, content_ids, codes, speaker, language='chinese'):
    """Build aligned text, speaker and codec embeddings for teacher forcing."""
    talker = model.talker
    config = model.config
    device = talker.device
    role_ids, content_ids, codes = (v.to(device) for v in (role_ids, content_ids, codes))
    speaker = speaker.to(device=device, dtype=talker.get_input_embeddings().weight.dtype).reshape(1, 1, -1)
    projection = lambda ids: talker.text_projection(talker.get_text_embeddings()(ids))
    special = torch.tensor([[config.tts_pad_token_id, config.tts_bos_token_id,
                             config.tts_eos_token_id]], device=device)
    pad, bos, eos = projection(special).split(1, dim=1)
    tc = config.talker_config
    language = torch.tensor([[tc.codec_think_id, tc.codec_think_bos_id,
                              tc.codec_language_id[language], tc.codec_think_eos_id]], device=device)
    tags = torch.cat((talker.get_input_embeddings()(language), speaker,
                      talker.get_input_embeddings()(torch.tensor([[tc.codec_pad_id]], device=device))), dim=1)
    prefix = torch.cat((projection(role_ids.reshape(1, -1)),
                        torch.cat((pad.expand(1, 5, -1), bos), dim=1) + tags,
                        torch.cat((projection(content_ids.reshape(1, -1)), eos), dim=1)
                        + talker.get_input_embeddings()(torch.tensor([[tc.codec_pad_id]], device=device)),
                        pad + talker.get_input_embeddings()(torch.tensor([[tc.codec_bos_id]], device=device))), dim=1)
    frames = talker.get_input_embeddings()(codes[:, 0]).unsqueeze(0)
    for group in range(1, 16):
        frames = frames + talker.code_predictor.get_input_embeddings()[group - 1](codes[:, group]).unsqueeze(0)
    inputs = torch.cat((prefix, frames + pad), dim=1)
    labels = torch.cat((codes[:, 0], torch.tensor([tc.codec_eos_token_id], device=device)))
    return inputs, prefix.shape[1], codes, labels


def codec_distributions(model, sample, speaker, chunk_frames=32, language='chinese'):
    """Compute all codec-group next-token logits with explicit frame alignment."""
    inputs, prefix_length, codes, labels = build_inputs(model, sample['role_ids'], sample['content_ids'],
                                                       sample['codes'], speaker, language)
    attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
    positions = torch.arange(inputs.shape[1], device=inputs.device).view(1, 1, -1).expand(3, 1, -1)
    # No labels passed to Transformers: it must not shift these explicit targets again.
    hidden = model.talker.model(inputs_embeds=inputs, attention_mask=attention, position_ids=positions,
                                use_cache=False, output_hidden_states=False).last_hidden_state[0]
    prediction_hidden = hidden[prefix_length - 1:]
    first = model.talker.codec_head(prediction_hidden)
    if first.shape[0] != codes.shape[0] + 1:
        raise RuntimeError('主音频单元的预测与目标没有正确对齐。')
    predictor = model.talker.code_predictor
    sub_logits = []
    for start in range(0, len(codes), chunk_frames):
        current_codes = codes[start:start + chunk_frames]
        pieces = [prediction_hidden[start:start + len(current_codes)].unsqueeze(1),
                  model.talker.get_input_embeddings()(current_codes[:, :1])]
        for group in range(1, 15):
            pieces.append(predictor.get_input_embeddings()[group - 1](current_codes[:, group:group + 1]))
        sub_inputs = predictor.small_to_mtp_projection(torch.cat(pieces, dim=1))
        sub_hidden = predictor.model(inputs_embeds=sub_inputs, use_cache=False,
                                      output_hidden_states=False).last_hidden_state
        # Position 1 (codec-0 input) predicts codec-1; each later head predicts its own codebook.
        sub_logits.append(torch.stack([predictor.lm_head[group - 1](sub_hidden[:, group])
                                       for group in range(1, 16)], dim=1))
    return first, torch.cat(sub_logits, dim=0), labels, codes[:, 1:]


def distribution_loss(student_logits, teacher_logits, targets, temperature, kd_weight):
    """Compute real-code cross entropy and temperature-scaled teacher KL loss."""
    s = student_logits.float().reshape(-1, student_logits.shape[-1])
    t = teacher_logits.detach().to(device=s.device, dtype=torch.float32).reshape_as(s)
    ce = F.cross_entropy(s, targets.reshape(-1))
    kd = F.kl_div(F.log_softmax(s / temperature, dim=-1),
                  F.softmax(t / temperature, dim=-1), reduction='batchmean') * temperature ** 2
    return (1 - kd_weight) * ce + kd_weight * kd, ce, kd
