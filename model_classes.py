import torch
import torch.nn as nn
import torch.nn.functional as F


class SurveyCrossAttention(nn.Module):
    """
    EHR hidden states are queries.
    Survey hidden states are keys/values.

    ehr_hidden_states:
        [batch, ehr_seq_len, ehr_dim]

    survey_hidden_states:
        [batch, survey_seq_len, survey_dim]
    """

    def __init__(
        self,
        ehr_dim: int,
        survey_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        cross_attn_scale_init: float = 0.1,
    ):
        super().__init__()

        if ehr_dim % num_heads != 0:
            raise ValueError(
                f"ehr_dim={ehr_dim} must be divisible by "
                f"num_heads={num_heads}"
            )

        self.survey_proj = nn.Linear(
            survey_dim,
            ehr_dim,
            bias=False,
        )

        self.ehr_norm = nn.LayerNorm(ehr_dim)
        self.survey_norm = nn.LayerNorm(ehr_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=ehr_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.cross_attn_scale = nn.Parameter(
            torch.tensor(float(cross_attn_scale_init))
        )

    def forward(
        self,
        ehr_hidden_states,
        survey_hidden_states,
        ehr_attention_mask=None,
        survey_attention_mask=None,
    ):
        residual = ehr_hidden_states

        # EHR = queries
        q = self.ehr_norm(ehr_hidden_states)

        # Survey = keys/values
        survey_hidden_states = self.survey_proj(
            survey_hidden_states
        )
        survey_hidden_states = self.survey_norm(
            survey_hidden_states
        )

        key_padding_mask = None

        if survey_attention_mask is not None:
            # MultiheadAttention: True means ignore the position.
            key_padding_mask = ~survey_attention_mask.bool()

        attn_output, _ = self.cross_attn(
            query=q,
            key=survey_hidden_states,
            value=survey_hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        if ehr_attention_mask is not None:
            attn_output = (
                attn_output
                * ehr_attention_mask.unsqueeze(-1).to(
                    attn_output.dtype
                )
            )

        hidden_states = (
            residual
            + self.cross_attn_scale
            * self.dropout(attn_output)
        )

        return hidden_states


class SurveyMambaBlock(nn.Module):
    """
    Pretrained Mamba block -> optional survey cross-attention -> next block.
    """

    def __init__(
        self,
        mamba_block,
        ehr_dim: int,
        survey_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_cross_attn: bool = True,
    ):
        super().__init__()

        self.mamba_block = mamba_block
        self.use_cross_attn = use_cross_attn

        if self.use_cross_attn:
            self.survey_cross_attn = SurveyCrossAttention(
                ehr_dim=ehr_dim,
                survey_dim=survey_dim,
                num_heads=num_heads,
                dropout=dropout,
            )

    def forward(
        self,
        hidden_states,
        survey_hidden_states,
        ehr_attention_mask=None,
        survey_attention_mask=None,
    ):
        hidden_states = self.mamba_block(
            hidden_states,
            cache_params=None,
            attention_mask=ehr_attention_mask,
        )

        if self.use_cross_attn:
            hidden_states = self.survey_cross_attn(
                ehr_hidden_states=hidden_states,
                survey_hidden_states=survey_hidden_states,
                ehr_attention_mask=ehr_attention_mask,
                survey_attention_mask=survey_attention_mask,
            )

        return hidden_states

class GeneHeadLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_selected_layers,
        num_genes,
        gene_embedding_dim=32,
    ):
        super().__init__()

        self.num_genes = num_genes
        self.gene_embedding_dim = gene_embedding_dim

        input_dim = hidden_dim * num_selected_layers

        self.gene_weights = nn.Parameter(
            torch.empty(
                num_genes,
                input_dim,
                gene_embedding_dim,
            )
        )

        self.gene_bias = nn.Parameter(
            torch.zeros(
                num_genes,
                gene_embedding_dim,
            )
        )

        self.output_weights = nn.Parameter(
            torch.empty(
                num_genes,
                gene_embedding_dim,
            )
        )

        self.output_bias = nn.Parameter(
            torch.zeros(num_genes)
        )

        nn.init.xavier_uniform_(self.gene_weights)
        nn.init.xavier_uniform_(self.output_weights)

    def forward(self, selected_embeddings):
        # [B, num_selected_layers * hidden_dim]
        x = torch.cat(
            selected_embeddings,
            dim=-1,
        )

        # [B, G, E]
        gene_embeddings = torch.einsum(
            "bd,gde->bge",
            x,
            self.gene_weights,
        )

        gene_embeddings = (
            gene_embeddings
            + self.gene_bias.unsqueeze(0)
        )

        gene_embeddings = torch.tanh(
            gene_embeddings
        )

        # [B, G]
        gene_logits = torch.einsum(
            "bge,ge->bg",
            gene_embeddings,
            self.output_weights,
        )

        gene_logits = (
            gene_logits
            + self.output_bias.unsqueeze(0)
        )

        return gene_logits


class GermLinePredModel(nn.Module):
    def __init__(
        self,
        survey_model,
        ehr_model,
        xai_layers,
        num_selected_layers,
        num_genes,
        gene_embedding_dim=32,
        attn_heads=8,
        dropout=0.1,
        pos_weight=None,
    ):
        super().__init__()

        self.xai_layers = list(xai_layers)
        self.num_selected_layers = num_selected_layers
        self.num_genes = num_genes

        self.survey_model = survey_model
        self.ehr_model = ehr_model

        ehr_dim = self.ehr_model.config.hidden_size
        survey_dim = self.survey_model.config.hidden_size

        # PeftModel.base_model keeps the LoRA-wrapped base model.
        self.survey_encoder = self.survey_model.base_model

        self.ehr_embeddings = (
            self.ehr_model.backbone.embeddings
        )

        self.final_norm = (
            self.ehr_model.backbone.norm_f
        )

        original_layers = list(
            self.ehr_model.backbone.layers
        )

        self.layers = nn.ModuleList(
                        [
                            SurveyMambaBlock(
                                mamba_block=layer,
                                ehr_dim=ehr_dim,
                                survey_dim=survey_dim,
                                num_heads=attn_heads,
                                dropout=dropout,
                                use_cross_attn=((layer_idx + 1) % 6 == 0),
                            )
                            for layer_idx, layer in enumerate(original_layers)
                        ]
                    )

        self.prediction_head = nn.Linear(
            ehr_dim,
            1,
        )

        self.gene_pred_layer = GeneHeadLayer(
            hidden_dim=ehr_dim,
            num_selected_layers=num_selected_layers,
            num_genes=num_genes,
            gene_embedding_dim=gene_embedding_dim,
        )

        if pos_weight is not None:
            self.register_buffer(
                "pos_weight",
                torch.as_tensor(
                    pos_weight,
                    dtype=torch.float32,
                ),
            )
        else:
            self.pos_weight = None

    def compress_survey(
        self,
        survey_hidden_states,
        survey_attention_mask,
        target_length=128,
    ):
        """
        Compress survey sequence from [B, L, D] to [B, target_length, D]
        using adaptive average pooling over valid (non-padding) tokens.
        """
    
        batch_size, _, hidden_dim = survey_hidden_states.shape
    
        compressed_states = []
        compressed_masks = []
    
        for i in range(batch_size):
    
            valid_length = int(
                survey_attention_mask[i].sum().item()
            )
    
            x = survey_hidden_states[
                i,
                :valid_length,
                :
            ]
    
            if valid_length > target_length:
    
                # [L, D] -> [1, D, L]
                x = x.transpose(0, 1).unsqueeze(0)
    
                x = F.adaptive_avg_pool1d(
                    x,
                    target_length,
                )
    
                # [1, D, target_length]
                # -> [target_length, D]
                x = x.squeeze(0).transpose(0, 1)
    
                mask = torch.ones(
                    target_length,
                    dtype=survey_attention_mask.dtype,
                    device=survey_attention_mask.device,
                )
    
            else:
    
                padding_length = (
                    target_length - valid_length
                )
    
                x = F.pad(
                    x,
                    (0, 0, 0, padding_length),
                )
    
                mask = torch.cat(
                    [
                        torch.ones(
                            valid_length,
                            dtype=survey_attention_mask.dtype,
                            device=survey_attention_mask.device,
                        ),
                        torch.zeros(
                            padding_length,
                            dtype=survey_attention_mask.dtype,
                            device=survey_attention_mask.device,
                        ),
                    ]
                )
    
            compressed_states.append(x)
            compressed_masks.append(mask)
    
        return (
            torch.stack(compressed_states),
            torch.stack(compressed_masks),
        )

        
    def forward(
        self,
        ehr_input_ids,
        ehr_attention_mask,
        survey_input_ids,
        survey_attention_mask,
        labels=None,
    ):
        if not hasattr(self, "_printed_shapes"):
            print(
                "EHR shape:",
                ehr_input_ids.shape,
                "| Survey shape:",
                survey_input_ids.shape,
                flush=True,
            )
            self._printed_shapes = True
        # =========================================
        # Survey encoder
        # =========================================
        survey_outputs = self.survey_encoder(
            input_ids=survey_input_ids,
            attention_mask=survey_attention_mask,
            return_dict=True,
        )

        survey_hidden_states = (
            survey_outputs.last_hidden_state
        )

        # survey_hidden_states, survey_attention_mask = (
        #     self.compress_survey(
        #             survey_hidden_states,
        #             survey_attention_mask,
        #             target_length=128,
        #     )
        # )

        # =========================================
        # EHR embeddings
        # =========================================
        hidden_states = self.ehr_embeddings(
            ehr_input_ids
        )

        # Last valid event for each patient.
        patient_lengths = (
            ehr_attention_mask.long().sum(dim=1)
        )

        final_event_indices = patient_lengths - 1

        batch_indices = torch.arange(
            hidden_states.shape[0],
            device=hidden_states.device,
        )

        # =========================================
        # Mamba layers + survey cross-attention
        # =========================================
        selected_embeddings = []

        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
                survey_hidden_states=survey_hidden_states,
                ehr_attention_mask=ehr_attention_mask,
                survey_attention_mask=survey_attention_mask,
            )

            if layer_idx in self.xai_layers:
                layer_embedding = hidden_states[
                    batch_indices,
                    final_event_indices,
                ]

                selected_embeddings.append(
                    layer_embedding
                )

        # =========================================
        # Final Mamba normalization
        # =========================================
        hidden_states = self.final_norm(
            hidden_states
        )

        patient_embedding = hidden_states[
            batch_indices,
            final_event_indices,
        ]

        final_layer_idx = len(self.layers) - 1

        if final_layer_idx in self.xai_layers:
            xai_idx = self.xai_layers.index(
                final_layer_idx
            )

            selected_embeddings[
                xai_idx
            ] = patient_embedding

        # =========================================
        # Predictions
        # =========================================
        overall_logit = self.prediction_head(
            patient_embedding
        )

        gene_logits = self.gene_pred_layer(
            selected_embeddings
        )

        logits = torch.cat(
            (
                overall_logit,
                gene_logits,
            ),
            dim=1,
        )

        # =========================================
        # Loss
        # =========================================
        loss = None
        overall_loss = None
        gene_loss = None

        if labels is not None:
            labels = labels.float()

            overall_labels = labels[:, 0]
            overall_logits_flat = overall_logit.squeeze(-1)

            if self.pos_weight is not None:
                overall_pos_weight = self.pos_weight[0]
            else:
                overall_pos_weight = None

            overall_loss = F.binary_cross_entropy_with_logits(
                overall_logits_flat,
                overall_labels,
                pos_weight=overall_pos_weight,
            )

            gene_labels = labels[:, 1:]

            if self.pos_weight is not None:
                gene_pos_weight = self.pos_weight[1:]
            else:
                gene_pos_weight = None

            gene_loss = F.binary_cross_entropy_with_logits(
                gene_logits,
                gene_labels,
                pos_weight=gene_pos_weight,
            )

            loss = (
                0.5 * overall_loss
                + 0.5 * gene_loss
            )

        return {
            "loss": loss,
            "logits": logits,
            "overall_loss": overall_loss,
            "gene_loss": gene_loss,
        }
