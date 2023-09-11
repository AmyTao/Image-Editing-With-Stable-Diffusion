# Terms:

* Schduler
1. determine the noise level at each steps
example: DDPM, LMS (different startegies in controlling the noise level)
2. related with the coherence, generation speed,and quality of generated images.

* Diffuser
1. The one adds/removes noise (assigned by scheduler) in the generation process.

* noise residual:
The difference between two images in sequence.

* likelihood: how well the model fit the data (a prob)

* KL divergence: how a prob distribution different from another prob distribution

* down_block in Unet

* up_block in Unet

* text embedding: tockenizer preprocess the data (numerical expression of input tokens)

# Diffusion models Structure

* Goal: learn the distribution of data, by learning reverse process

* Generate data by sampling from the distribution

* Adding noise:beta_t control the noise level (assigned by scheduler)

* Challenge: forwarding process add Gaussain noise in each step, but we don't know the prob of the reverse process

=> we approximate that it should also be a Gaussian distribution

=> now the goal turns to be learning the mean and variance

[explain the process](https://www.sohu.com/a/660579806_121438385)

- useful function
```
import PIL.Image
import numpy as np


def display_sample(sample, i):
    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    display(f"Image at step {i}")
    display(image_pil)


import tqdm

sample = noisy_sample

for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
    # 1. predict noise residual
    with torch.no_grad():
        residual = model(sample, t).sample

    # 2. compute less noisy image and set x_t -> x_t-1
    sample = scheduler.step(residual, t, sample).prev_sample

    # 3. optionally look at image
    if (i + 1) % 50 == 0:
        display_sample(sample, i + 1)
```

- logic flow of StableDiffusion
<img src="/pics/stable_diffusion.png" alt="图片alt" title="图片title">