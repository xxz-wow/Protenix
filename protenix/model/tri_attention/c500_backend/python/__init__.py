"""C500 tri_attention 前向与一阶反向接口。"""
from .api import Backend, attention, attention_raw, TriAttention, TriAttentionFunction

__all__ = ["TriAttention", "TriAttentionFunction", "Backend", "attention", "attention_raw"]
