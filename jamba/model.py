import torch
from torch import Tensor, nn
from zeta import MambaBlock
from zeta.nn import FeedForward
from zeta import MultiQueryAttention
from zeta.nn.modules.simple_rmsnorm import SimpleRMSNorm
from jamba.moe import MoE


class TransformerMoEBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        num_experts: int,
        num_experts_per_token: int,
        *args,
        **kwargs,
    ):
        """
        Initializes a TransformerMoEBlock.

        Args:
            dim (int): The dimension of the input tensor.
            heads (int): The number of attention heads.
            num_experts (int): The total number of experts.
            num_experts_per_token (int): The number of experts per token.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_token

        self.attn = MultiQueryAttention(dim, heads)
        self.moe = MoE(
            dim,
            num_experts=num_experts,
            hidden_dim=dim * 4,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the TransformerMoEBlock.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the TransformerMoEBlock.
        """
        skip = x
        x = SimpleRMSNorm(self.dim)(x)
        x, _, _ = self.attn(x) + x

        x = SimpleRMSNorm(self.dim)(x)
        moe_out, _ = self.moe(x)
        x = moe_out + skip
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        *args,
        **kwargs,
    ):
        """
        Initializes a TransformerBlock.

        Args:
            dim (int): Dimension of the input tensor.
            heads (int): Number of attention heads.
            num_experts (int): Number of experts.
            num_experts_per_token (int): Number of experts per token.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.dim = dim
        self.heads = heads

        self.attn = MultiQueryAttention(dim, heads)
        self.ffn = FeedForward(
            dim,
            dim,
            4,
            swish=True,
            post_act_ln=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the TransformerBlock.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the TransformerBlock.
        """
        skip = x
        x = SimpleRMSNorm(self.dim)(x)
        x, _, _ = self.attn(x)
        x += skip

        skip_two = x

        x = SimpleRMSNorm(self.dim)(x)
        x = self.ffn(x) + skip_two
        return x


class MambaMoELayer(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        d_conv: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        *args,
        **kwargs,
    ):
        """
        Initialize the MambaMoELayer.

        Args:
            dim (int): Dimension of the input tensor.
            d_state (int): Dimension of the state tensor.
            d_conv (int): Dimension of the convolutional tensor.
            num_experts (int, optional): Number of experts. Defaults to 8.
            num_experts_per_token (int, optional): Number of experts per token. Defaults to 2.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_token

        # Mamba
        self.mamba = MambaBlock(
            dim,
            depth=1,
            d_state=d_state,
            d_conv=d_conv,
        )

        # MoE
        self.moe = MoE(
            dim,
            num_experts=num_experts,
            hidden_dim=dim * 4,
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the MambaMoELayer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the MambaMoELayer.
        """
        skip = x

        x = SimpleRMSNorm(self.dim)(x)
        x = self.mamba(x) + x

        x = SimpleRMSNorm(self.dim)(x)
        moe_out, _ = self.moe(x)
        x = moe_out + skip
        return x


class JambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        d_conv: int,
        heads: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.heads = heads
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_token

        # Mamba
        self.mamba_layer = MambaBlock(
            dim,
            depth=1,
            d_state=d_state,
            d_conv=d_conv,
        )

        # Mamba MoE layer
        self.mamba_moe_layer = MambaMoELayer(
            dim,
            d_state,
            d_conv,
            num_experts,
            num_experts_per_token,
        )

        # Transformer
        self.transformer = TransformerBlock(
            dim,
            heads,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.mamba_layer(x)
        x = self.mamba_moe_layer(x)
        x = self.transformer(x)
        x = self.mamba_moe_layer(x)
        x = self.mamba_layer(x)
        x = self.mamba_moe_layer(x)
        return x


