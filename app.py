# import gradio as gr
import argparse
import itertools
import math
import os
import random

import requests
from io import BytesIO

import sys

import accelerate

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import PIL
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer


def training_function(text_encoder, vae, unet, hyperparameters, train_dataset, tokenizer, noise_scheduler, placeholder_token_id, placeholder_token):
    train_batch_size = hyperparameters["train_batch_size"]
    gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
    learning_rate = hyperparameters["learning_rate"]
    max_train_steps = hyperparameters["max_train_steps"]
    output_dir = hyperparameters["output_dir"]

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        # distributed_type='MULTI_GPU' 
        # distribute_type: Accelerate.DistributedType.MULTI_GPU
        # fp16=True,
        # cpu=True,
    )

    train_dataloader = create_dataloader(train_dataset, train_batch_size)

    if hyperparameters["scale_lr"]:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=learning_rate,
    )


    text_encoder, optimizer, train_dataloader = accelerator.prepare(
        text_encoder, optimizer, train_dataloader
    )

    # Move vae and unet to device
    vae.to(accelerator.device)
    unet.to(accelerator.device)

    # Keep vae and unet in eval model as we don't train these
    vae.eval()
    unet.eval()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample().detach()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)

                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if accelerator.num_processes > 1:
                    grads = text_encoder.module.get_input_embeddings().weight.grad
                else:
                    grads = text_encoder.get_input_embeddings().weight.grad
                # Get the index for tokens that we want to zero the grads for
                index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
                grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)

                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        accelerator.wait_for_everyone()


    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline(
            text_encoder=accelerator.unwrap_model(text_encoder),
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            scheduler=PNDMScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True, steps_offset=1
            ),
            safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
        )
        pipeline.save_pretrained(output_dir)
        # Also save the newly trained embeddings
        learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
        learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
        torch.save(learned_embeds_dict, os.path.join(output_dir, "learned_embeds.bin"))


#@title Setup the dataset
class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):

        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": Image.Resampling.BILINEAR,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example

def freeze_params(params):
    for param in params:
        param.requires_grad = False

def create_dataloader(train_dataset, train_batch_size=16):
    return torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

# prevent safety checking
def dummy(images, **kwargs):
    return images, False


# ---------------------------

#@title Setup the prompt templates for training 
imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

def train2images(params, isTrain):

    initializer_token = "groot"
    concept_name = params["hash"];
    placeholder_token = f'<{concept_name}>'

    
    
    if (os.path.isdir("in-data/" + params["hash"])):
        in_path = "in-data/" + params["hash"]
    else:
        in_path = "inputs/wavediana-1"
    
    out_path = "out-data/" + params["hash"]
    out_model_path = "model-data/" + params["hash"]


    if not os.path.exists(out_path):
        os.mkdir(out_path)

    if not os.path.exists(out_model_path):
        os.mkdir(out_model_path)


    if (isTrain):

        pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
        # pretrained_model_name_or_path = "v1-5-pruned.ckpt"
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        tokenizer.add_tokens(placeholder_token)
        token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)

        initializer_token_id = token_ids[0]
        placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder",

        )
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae",

        )
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet",

        )

        text_encoder.resize_token_embeddings(len(tokenizer))

        token_embeds = text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

        # Freeze vae and unet
        freeze_params(vae.parameters())
        freeze_params(unet.parameters())
        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        freeze_params(params_to_freeze)

        train_dataset = TextualInversionDataset(
            data_root=in_path,
            tokenizer=tokenizer,
            size=512,
            placeholder_token=placeholder_token,
            repeats=100,
            learnable_property=params["what_to_teach"],
            center_crop=False,
            set="train",
        )

        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=params["train_steps"]
        )

        hyperparameters = {
            "learning_rate": 5e-04,
            "scale_lr": True,
            "max_train_steps": params["train_steps"],
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "seed": 42,
            "output_dir": out_model_path,
        }

        accelerate.notebook_launcher(training_function, args=(text_encoder, vae, unet, hyperparameters, train_dataset, tokenizer, noise_scheduler, placeholder_token_id, placeholder_token), num_processes=1)

    pipe = StableDiffusionPipeline.from_pretrained(
        out_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    ).to("cuda")

    pipe.safety_checker = dummy

    all_images = [] 
    for _ in range(params["num_rows"]):
        images = pipe(prompt=placeholder_token + " " + params["prompt"], negative_prompt=params["negative"], num_inference_steps=params["num_inference_steps"], guidance_scale=params["guidance_scale"]).images
        all_images.extend(images)


    # [image.save(f"{out_path}/{i}.jpeg") for i, image in enumerate(all_images)]
    return all_images

# sendParams = {
#     "hash": "Jr8OvhM9aNemx0Gnw1NmPRIgjxEmT8cx",
#     "train_steps": 500,
#     "what_to_teach": "object",
#     "prompt": "portrait, elegant, cyber neon lights, in a cyberpunk environment, highly detailed, digital painting, trending on artstation, trending on pinterest, concept art, sharp focus, centered, 8k, artgerm, greg rutkowski, mucha",
#     "negative": "cropped, out of frame, ugly, long neck, mutation, blurry, gross proportions, ((poorly drawn eyes)), duplicate, ((bad anatomy)), mutilated, poorly drawn face, poorly drawn hands, extra fingers, deformed, cloned face, deformed mouth, duplicated face features, extra ears, low resolution",
#     "num_samples": 5,
#     "num_rows": 10
# }

# train2images(sendParams, True)


if not os.path.exists("out-data"):
    os.mkdir("out-data")

if not os.path.exists("model-data"):
    os.mkdir("model-data")

import gradio as gr

def doAction(images, model_name_train, model_name_trained, train_steps, prompt, negative, num_images, num_inference_steps, guidance_scale):
  
  if (model_name_train != "" and images):
    if not os.path.exists(f"in-data/{model_name_train}"):
        os.mkdir(f"in-data/{model_name_train}")
    for i, image in enumerate(images):
        new_file = open(f"in-data/{model_name_train}/{i}.jpeg", 'rw')
        shutil.copyfileobj(image, new_file)
        # loadImage = Image.open(BytesIO(image)).convert("RGB")
        # loadImage.save(f"in-data/{model_name_train}/{i}.jpeg")
  
  if (model_name_train != ""):
    return train2images({
        "hash": model_name_train,
        "train_steps": train_steps,
        "what_to_teach": "object",
        "prompt": prompt,
        "negative": negative,
        "num_samples": 5,
        "num_rows": num_images,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale
    }, True)
  else:
    return train2images({
        "hash": model_name_trained,
        "train_steps": train_steps,
        "what_to_teach": "object",
        "prompt": prompt,
        "negative": negative,
        "num_samples": 5,
        "num_rows": num_images,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale
    }, False)

app = gr.Interface(
  doAction, 
  [
      gr.File(file_count="multiple", file_types="image"),
      gr.Textbox(value=""),
      gr.Textbox(value=""),
      gr.Slider(100, 1000, value=500),
      gr.Textbox(value="portrait, elegant, cyber neon lights, in a cyberpunk environment, highly detailed, digital painting, trending on artstation, trending on pinterest, concept art, sharp focus, centered, 8k, artgerm, greg rutkowski, mucha"),
      gr.Textbox(value="cropped, out of frame, ugly, long neck, mutation, blurry, gross proportions, ((poorly drawn eyes)), duplicate, ((bad anatomy)), mutilated, poorly drawn face, poorly drawn hands, extra fingers, deformed, cloned face, deformed mouth, duplicated face features, extra ears, low resolution"),
      gr.Slider(2, 20, value=5),
      gr.Slider(10, 100, value=50),
      gr.Slider(1, 20, value=7.5),
  ],
  [
    gr.Gallery()
  ]
)

app.queue(concurrency_count=1)
app.launch(server_name="0.0.0.0", debug=True, share=True)
