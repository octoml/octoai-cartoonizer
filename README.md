# OctoAI: Cartoonizer Tutorial

In this guide you'll learn how to build, deploy and share your own interactive and engaging image-to-image Generative AI web-application using OctoAI!


[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tmoreau89-cartoonize-cartoonizer-gwpv6p.streamlit.app/)

## Requirements

Let's take a look at the requirements in terms of skill set, hardware and software for each phase of this tutorial.

In total this tutorial takes 1-2 hours depending on how much programming experience you have and how much you want to experiment with stable diffusion functionality. Note that we've designed this tutorial to be as approachable as possible. Very minimal programming experience is required here to be successful.

### Phase 1: Experiment with Image to Image feature in Stable Diffusion Web UI

#### Time
* 30-60mins depending on how much you want to experiment with Stable Diffusion.

#### Skillset
* Git, and beginer-level command line programming.

#### Hardware
* A computer of instance with a beefy GPU (Nvidia A10 or better if you want to run the CLIP interrogator model to perform image to text).

#### Software
* [AUTOMATIC1111 stable diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) which is under AGPL-3.0 license.
* A model checkpoint of your choice from [CivitAI's website](https://civitai.com/).
* An image of your choice.


### Phase 2: Build your own CLIP-Interrogator Docker container

#### Time
* 15-30 mins.

#### Skillset
* Git and beginner-level command line and Python programming.

#### Hardware
* A computer of instance with a beefy GPU (Nvidia A10 or better if you want to run the CLIP interrogator model to perform image to text).

#### Software
* [Docker Engine](https://docs.docker.com/engine/) and a [Docker Hub](https://hub.docker.com/) account.
* [pharmapscychotic clip interrogator library](https://github.com/pharmapsychotic/clip-interrogator) which available under MIT license.
* The ["Build a Container from Python" guide](https://docs.octoai.cloud/docs/create-custom-endpoints-from-python-code) from OctoAI's website.

### Phase 3: Deploy your inference REST endpoint

#### Time
* 5-15 mins.

#### Skillset
* No experience needed to launch the OctoAI endpoint. Beginner-level command line programming for testing.

#### Hardware
* Your laptop.

#### Software
* [OctoAI's compute service](https://docs.octoai.cloud/docs) with a user account.
* The CLIP-Interrogator model container built in Phase 2 and uploaded on [Docker Hub](https://hub.docker.com/).
* A Stable Diffusion model container that we built for you.
* A test script to exercise your newly-launched OctoAI endpoint.s

### Phase 4: Build your own Streamlit Web Frontend

#### Time
* 15-30 mins.

#### Skillset
* No web-design experience needed! Beginner-level Python programming.

#### Hardware
* Your laptop for developing your webapp, a laptop or phone/tablet to test the webapp.

#### Software
* [Streamlit](https://streamlit.io/) library that you can pip-install.
* The OctoAI inference endpoint URLs that you launched in Phase 3.

## Disclaimer

Under construction.

## Step-by-step Cartoonizer Tutorial

### Phase 1: Experiment with Image to Image feature in Stable Diffusion Web UI

Under construction.

### Phase 2: Build your own CLIP-Interrogator Docker container

Let's walk through how you can build your own model container and upload it on DockerHub. Once uploaded you can use OctoAI's compute service to serve the model container behind an easy to use REST endpoint. This makes your model easy to access, manage and scale based on your use requirements.

#### A. Test out CLIP-Interrogator

As we saw in Phase 1, image to image generation depends on being able to generate a prompt from an input image. This promp will be used to guide the generation of the output "cartoonized" image.

For this exercise, we'll use [pharmapscychotic clip interrogator library](https://github.com/pharmapsychotic/clip-interrogator) which available under MIT license.

We follow the instructions available to their website. Namely from a computer/instance that you can program on, open a terminal (or ssh into it) and do the following.

Create and activate a python virtual environment. This will create a new local environment where you can install pip packages without needing to worry about breaking dependencies for your other Python projects.

```
cd <MY CLIP TEST DIRECTORY>
python3 -m venv ci_env
(for linux  ) source ci_env/bin/activate
(for windows) .\ci_env\Scripts\activate
```

Now install with pip.
```
# install torch with GPU support for example:
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117

# install clip-interrogator and Pillow (image library)
pip install clip-interrogator==0.5.4
pip install Pillow
```

You are ready to test out CLIP on an image of your choice with the following python test script that you can save in `test_clip.py`. Replace `image_path` below to point to your input image.
```
from PIL import Image
from clip_interrogator import Config, Interrogator

image = Image.open(image_path).convert('RGB')
ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
print(ci.interrogate(image))
```

#### B. Familiarize yourself with OctoAI's guide to building a model container

We're going to use the [Build a Container from Python Guide](https://docs.octoai.cloud/docs/create-custom-endpoints-from-python-code) as a guide to build our own Clip-Interrogator model.

### Phase 3: Deploy your inference REST endpoint

Under construction.

### Phase 4: Build your own Streamlit Web Frontend

Under construction.

