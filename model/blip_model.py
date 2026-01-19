import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.functional import normalize
from transformers import BlipPreTrainedModel, BlipConfig, BlipVisionModel, BlipTextModel

class BlipForImageTextRetrievalVision(BlipPreTrainedModel):
    config_class = BlipConfig

    def __init__(self, config: BlipConfig):
        super().__init__(config)

        self.vision_model = BlipVisionModel(config.vision_config)

        # vision projection layer
        self.vision_proj = nn.Linear(config.vision_config.hidden_size, config.image_text_hidden_size)

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(self,  
                pixel_values: torch.FloatTensor,        
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                interpolate_pos_encoding: bool = False):
      
      vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
      image_embeds = vision_outputs[0]
      image_feat = normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
      return image_feat
    
class BlipForImageTextRetrievalText(BlipPreTrainedModel):
    config_class = BlipConfig

    def __init__(self, config: BlipConfig):
        super().__init__(config)

        self.text_encoder = BlipTextModel(config.text_config, add_pooling_layer=False)

        # text projection layer
        self.text_proj = nn.Linear(config.text_config.hidden_size, config.image_text_hidden_size)

        self.decoder_pad_token_id = (
            config.text_config.pad_token_id
            if not hasattr(config, "decoder_pad_token_id")
            else config.decoder_pad_token_id
        )
        self.decoder_start_token_id = (
            config.text_config.bos_token_id
            if not hasattr(config, "decoder_start_token_id")
            else config.decoder_start_token_id
        )

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(self,  
                input_ids: torch.LongTensor,        
                attention_mask: Optional[torch.LongTensor] = None,                
                return_dict: Optional[bool] = None):
      
      question_embeds = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
            )
      question_embeds = question_embeds[0] 
      text_feat = normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)
      return text_feat    


class BlipForImageTextRetrievalWrapper(BlipPreTrainedModel):
    config_class = BlipConfig

    def __init__(self, config: BlipConfig):
        super().__init__(config)

        self.vision_model = BlipVisionModel(config.vision_config)

        self.text_encoder = BlipTextModel(config.text_config, add_pooling_layer=False)

        # vision projection layer
        self.vision_proj = nn.Linear(config.vision_config.hidden_size, config.image_text_hidden_size)

        # text projection layer
        self.text_proj = nn.Linear(config.text_config.hidden_size, config.image_text_hidden_size)

        # image text matching head
        self.itm_head = nn.Linear(config.text_config.hidden_size, 2)

        self.decoder_pad_token_id = (
            config.text_config.pad_token_id
            if not hasattr(config, "decoder_pad_token_id")
            else config.decoder_pad_token_id
        )
        self.decoder_start_token_id = (
            config.text_config.bos_token_id
            if not hasattr(config, "decoder_start_token_id")
            else config.decoder_start_token_id
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        use_itm_head: Optional[bool] = True,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ):
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, BlipForImageTextRetrieval

        >>> model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "an image of a cat"

        >>> inputs = processor(images=image, text=text, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        image_embeds = vision_outputs[0]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

        if use_itm_head:
            question_embeds = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=return_dict,
            )
            question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state

            output = self.itm_head(question_embeds[:, 0, :])
        else:
            question_embeds = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
            )
            question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state

            image_feat = normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
            text_feat = normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)

            output = image_feat @ text_feat.t()
        
        outputs = (output, vision_outputs[0]) + vision_outputs[2:] + (question_embeds,)
        return tuple(output for output in outputs if output is not None)
      
    def encode_image(self, pixel_values: torch.FloatTensor):
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            interpolate_pos_encoding=False
        )
        image_embeds = vision_outputs[0]
        #image_feat = normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        image_feat = normalize(self.vision_proj(image_embeds), dim=-1)
        return image_feat
     
    def encode_text(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor] = None):
        question_embeds = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=None,
                )
        question_embeds = question_embeds[0] 
        #text_feat = normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)
        text_feat = normalize(self.text_proj(question_embeds), dim=-1)
        return text_feat

if __name__ == '__main__':
    from transformers import BlipProcessor, BlipModel
    from PIL import Image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'Salesforce/blip-itm-base-coco'    
    print('loading model...')
    visual_model = BlipForImageTextRetrievalVision.from_pretrained(model_name).to(device)
    text_model = BlipForImageTextRetrievalText.from_pretrained(model_name).to(device)
    processor = BlipProcessor.from_pretrained(model_name)       

    pil_img = Image.open('images/dog_man.jpg').convert('RGB')
    
    # Provide a list of text captions
    texts = ["man and dog", "a man is talking to a woman", "a cute dog is walking outside"]

    # 3. Process inputs
    # The processor prepares the image and text for the model.
    img_inputs = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
    text_inputs = processor(text=texts, return_tensors="pt", padding=True).input_ids.to(device)

    # 4. Get the image-text matching scores
    # The model returns a similarity score for each image-text pair.
    print('computing similarity...')
    image_features = visual_model(img_inputs)
    text_features = text_model(text_inputs)
    
    similarity = image_features @ text_features.t()
    print(similarity)