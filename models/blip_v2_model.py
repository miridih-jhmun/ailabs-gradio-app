from typing import Any, Optional, Tuple, Union

import torch
from torch import nn

from dataclasses import dataclass

from transformers import Blip2Model
from transformers.utils import ModelOutput, logging
from torch.nn import CrossEntropyLoss

logger = logging.get_logger(__name__)


@dataclass
class ModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(self[k] if k not in [] else getattr(self, k).to_tuple() for k in self.keys())


class BlipV2Model(Blip2Model):

    def __init__(self, config, num_labels):
        super().__init__(config)
        self.config = config
        self.logit_scale = torch.nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        self.num_labels = num_labels
        self.classifier = nn.Sequential(
            nn.Linear(1408, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.1),
            nn.Linear(512, self.num_labels),
        )
        self.post_init()

    def forward(self,
                pixel_values: torch.FloatTensor = None,
                labels: Optional[torch.LongTensor] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs) -> Union[Tuple, ModelOutput]:
        embedding_output = self.vision_model.embeddings(pixel_values)
        vision_outputs = self.vision_model.encoder(embedding_output)

        last_hidden_state = vision_outputs[0]
        last_hidden_state = self.vision_model.post_layernorm(last_hidden_state)
        pooled_output = last_hidden_state[:, 0, :]
        image_embeds = self.vision_model.post_layernorm(pooled_output)

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        logits = self.classifier(image_embeds)

        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return ModelOutput(
            loss=loss,
            logits=logits,
        )
