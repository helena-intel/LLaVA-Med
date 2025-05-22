#!/usr/bin/env python
# coding: utf-8

# # Visual-language assistant with LLaVA Med and OpenVINO

# ## Prerequisites

# ## Get pretrained model

import gc
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import nncf
import numpy as np
import openvino as ov
import torch
from datasets import load_dataset
from openvino.runtime import opset13
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor

from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM, LlavaMistralModel

model_id = "microsoft/llava-med-v1.5-mistral-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = LlavaMistralForCausalLM.from_pretrained(model_id)

image_precision = "fp16"  # int8 or fp16
llm_precision = "int8"  # int4, int8 or fp32

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"

image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower)

mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
if mm_use_im_start_end:
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

vision_tower = model.get_vision_tower()


vision_config = vision_tower.config
vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
vision_config.use_im_start_end = mm_use_im_start_end
if mm_use_im_start_end:
    vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
    )


if hasattr(model.config, "max_sequence_length"):
    context_len = model.config.max_sequence_length
else:
    context_len = 2048


# ## Convert model


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


ov_out_path = Path(f'ov_llava_med_im{image_precision}_llm{llm_precision}')
model.config.save_pretrained(ov_out_path)

image_encoder_path = ov_out_path / "image_encoder.xml"
token_embedding_model_path = ov_out_path / "token_embed.xml"
second_stage_model_path = ov_out_path / "llava_with_past.xml"


# ### Image Encoder


class ImageEncoder(torch.nn.Module):
    def __init__(self, vision_tower, mm_projector):
        super().__init__()
        self.vision_tower = vision_tower
        self.vision_tower.load_model()
        self.mm_projector = mm_projector

    def forward(self, images):
        image_features = self.vision_tower(images)
        image_features = self.mm_projector(image_features)
        return image_features


if not image_encoder_path.exists():
    image_encoder = ImageEncoder(model.get_model().get_vision_tower(), model.get_model().mm_projector)
    with torch.no_grad():
        ov_model = ov.convert_model(image_encoder, example_input=torch.zeros((1, 3, 336, 336)), input=[(-1, 3, 336, 336)])
    ov.save_model(ov_model, image_encoder_path)
    cleanup_torchscript_cache()
    del ov_model
    gc.collect()
    print("Image Encoder model successfully converted")


# #### Apply quantization on Image Encoder


def prepare_calibration_data(dataloader, init_steps):
    """
    This function prepares calibration data from a dataloader for a specified number of initialization steps.
    It iterates over the dataloader, fetching batches and storing the relevant data.
    """
    data = []
    print(f"Fetching {init_steps} for the initialization...")
    counter = 0
    for batch in tqdm(dataloader):
        if counter == init_steps:
            break
        if batch:
            counter += 1
            with torch.no_grad():
                data.append(
                    {
                        "images": batch["images"].to("cpu"),
                    }
                )
    return data


def collate_fn(example, image_column="image"):
    """
    Preprocesses an example by loading and transforming image .
    Returns the preprocessed inputs with transformed image.
    """
    assert len(example) == 1
    example = example[0]
    image = example[image_column]
    h, w = image.size
    if h == 1 or w == 1:
        return None

    inputs = {}
    pixel_values = image_processor.preprocess(images=[image], return_tensors="pt")["pixel_values"]
    inputs["images"] = pixel_values
    return inputs


def prepare_dataset(opt_init_steps=300, max_train_samples=1000):
    """
    Prepares a vision-text dataset for quantization.
    """
    dataset = load_dataset("flaviagiammarino/vqa-rad", streaming=True)
    train_dataset = dataset["train"].shuffle(seed=42, buffer_size=max_train_samples)
    dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=1)
    calibration_data = prepare_calibration_data(dataloader, opt_init_steps)
    return calibration_data


if image_precision == "int8":
    print("Quantize Image Encoder")
    calibration_data = prepare_dataset()
    core = ov.Core()
    ov_image_encoder = core.read_model(image_encoder_path)
    calibration_dataset = nncf.Dataset(calibration_data)
    quantized_model = nncf.quantize(
        model=ov_image_encoder,
        calibration_dataset=calibration_dataset,
        model_type=nncf.ModelType.TRANSFORMER,
        # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
        advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.6),
    )

    with tempfile.TemporaryDirectory() as d:
        temp_path = Path(d) / "image_encoder.xml"
        ov.save_model(quantized_model, temp_path)
        os.replace(temp_path, image_encoder_path)
        os.replace(temp_path.with_suffix(".bin"), image_encoder_path.with_suffix(".bin"))

    print("Image encoder model successfully quantized")
    del ov_image_encoder
    del quantized_model
    gc.collect()


# ### Text Encoder

if not token_embedding_model_path.exists():
    ov_model = ov.convert_model(model.get_model().embed_tokens, example_input=torch.ones((1, 10), dtype=torch.long))
    ov.save_model(ov_model, token_embedding_model_path)
    cleanup_torchscript_cache()
    del ov_model
    gc.collect()
    print("Token Embedding model successfully converted")


# ### LLaMA


def model_has_state(ov_model: ov.Model):
    # TODO: Provide a better way based on the variables availability, but OV Python API doesn't expose required methods
    return len(ov_model.get_sinks()) > 0


def model_has_input_output_name(ov_model: ov.Model, name: str):
    """
    Helper function for checking that model has specified input or output name

    Parameters:
      ov_model (ov.Model):   # TODO: Can we derive the dimensions from the model topology?
      name (str):
          name of input or output

    Returns:
      True if input or output with requested name exists else False
    """
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])


def fuse_cache_reorder(ov_model: ov.Model, not_kv_inputs: List[str], key_value_input_names: List[str], gather_dim: int):
    """
    Fuses reored_cache during generate cycle into ov.Model. Used with stateful models, because we can not modify model state directly.

    Adds a new beam_idx parameter and Gather op per each kv-cache input in a given model.
    Should be run before make_stateful. Implements optimumum's _reorder_cache
    inside the model in the beginning of each iteration.
    Gather works along given gather_dim dimension that may vary from model to model.
    KV-cache inputs are identified based on names in key_value_input_names.
    Append the new beam_idx parameter to not_kv_inputs.

    Parameters:
      ov_model (`ov.Model`):
          openvino model for processing
      not_kv_inputs (`List[str]`):
          list of input nodes in model that not related to past key values
      key_value_input_names (`List[str]`):
          list of names for key value input layers
      gather_dim (int):
          dimension for gathering cache during reorder pass
    """

    if model_has_input_output_name(ov_model, "beam_idx"):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input("inputs_embeds").get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})  # why list is not accepted?
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])
    # Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = opset13.gather(parameter_output_port, beam_idx, opset13.constant(gather_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()


def build_state_initializer(ov_model: ov.Model, batch_dim: int):
    """
    Build initialization ShapeOf Expression for all ReadValue ops

    Parameters:
      ov_model (ov.Model):
          openvino model
      batch_dim (int):
          index of dimension corresponding to batch size
    """
    input_ids = ov_model.input("inputs_embeds")
    batch = opset13.gather(opset13.shape_of(input_ids, output_type="i64"), opset13.constant([0]), opset13.constant(0))
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [opset13.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim for dim in dims]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape)
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()


def make_stateful(
    ov_model: ov.Model,
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
    key_value_output_names: List[str],
    batch_dim: int,
    num_attention_heads: int,
    num_beams_and_batch: int = None,
):
    """
    Hides kv-cache inputs and outputs inside the model as variables.

    Parameters:
        ov_model (ov.Model):
            openvino model
        not_kv_inputs (`List[str]`):
            list of input nodes in model that not related to past key values
        key_value_input_names (`List[str]`):
            list of names for key value input layers
        key_value_output_names (`List[str]`):
            list of names for key value input layers
        batch_dim (int):
            index of batch dimension in key value layers
        num_attention_heads (int):
            number of attention heads for batch dimension initialization
        num_beams_an_batch (int):
            precalculated number of beams and batch for shapes initialization
    """
    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map = {}
    # TODO: Can we derive the dimensions from the model topology?

    if num_beams_and_batch is not None:
        # Set batch size for input_ids and attention mask to avoid dynamic dimension got propagated from the end of the model back to ReadValue
        for input in not_kv_inputs:
            shape = input.get_partial_shape()
            if shape.rank.get_length() <= 2:  # == 1 for beam_index
                shape[0] = num_beams_and_batch
                input.get_node().set_partial_shape(shape)
            else:
                # can be a warning
                raise ValueError(f"Rank of {input.get_any_name()} input of the model is not 2, batch size is not set")

    for kv_name_pair in zip(key_value_input_names, key_value_output_names):
        input_output_map[kv_name_pair[0]] = kv_name_pair[1]
        if num_beams_and_batch is not None:
            input = ov_model.input(kv_name_pair[0])
            shape = input.get_partial_shape()
            shape[batch_dim] = num_beams_and_batch * num_attention_heads
            input.get_node().set_partial_shape(shape)

    if num_beams_and_batch is not None:
        # Re-validation model if shapes are altered above
        ov_model.validate_nodes_and_infer_types()

    apply_make_stateful_transformation(ov_model, input_output_map)
    if num_beams_and_batch is None:
        build_state_initializer(ov_model, batch_dim)


def patch_stateful(ov_model):
    key_value_input_names = [key.get_any_name() for key in ov_model.inputs[1:]]
    key_value_output_names = [key.get_any_name() for key in ov_model.outputs[1:]]
    not_kv_inputs = [input for input in ov_model.inputs if not any(name in key_value_input_names for name in input.get_names())]
    if not key_value_input_names or not key_value_output_names:
        return
    batch_dim = 0
    num_attention_heads = 1

    fuse_cache_reorder(ov_model, not_kv_inputs, key_value_input_names, batch_dim)
    make_stateful(ov_model, not_kv_inputs, key_value_input_names, key_value_output_names, batch_dim, num_attention_heads, None)


class ModelWithPastWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.config.model_type = "mistral"
        # self.model.to_bettertransformer()
        self.mistral = super(LlavaMistralModel, model.model).forward

    def forward(self, inputs_embeds, past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None):
        outputs = self.mistral(inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=True)
        hidden_states = outputs[0]
        logits = self.model.lm_head(hidden_states)
        return logits, outputs.past_key_values


if not second_stage_model_path.exists():
    input_embeddings = model.model.embed_tokens(torch.ones((1, 10), dtype=torch.long))

    model_with_past = ModelWithPastWrapper(model)
    pkv = model_with_past(input_embeddings)[1]
    model_inputs = ["inputs_embeds"]
    model_outputs = ["logits"]
    for idx in range(len(pkv)):
        model_inputs.extend([f"past_key_values.{idx}.key", f"past_key_values.{idx}.value"])
        model_outputs.extend([f"present.{idx}.key", f"present.{idx}.value"])

    ov_model = ov.convert_model(model_with_past, example_input={"inputs_embeds": input_embeddings[:, -2:, :], "past_key_values": pkv})
    for input, input_name in zip(ov_model.inputs, model_inputs):
        input.get_tensor().set_names({input_name})
    for output, output_name in zip(ov_model.outputs, model_outputs):
        output.get_tensor().set_names({output_name})
    if make_stateful is not None:
        patch_stateful(ov_model)
    if llm_precision == "int8":
        llava_wc_parameters = dict(mode=nncf.CompressWeightsMode.INT8_ASYM)
    elif llm_precision == "int4":
        llava_wc_parameters = dict(mode=nncf.CompressWeightsMode.INT4_ASYM, group_size=128, ratio=0.8)
    if llm_precision != "fp32":
        print("Applying weight compression to second stage LLava model")
        ov_model = nncf.compress_weights(ov_model, **llava_wc_parameters)

    ov.save_model(ov_model, second_stage_model_path)
    cleanup_torchscript_cache()
    del ov_model
    gc.collect()
    print("Llava model successfully converted")
