# OctoAI: Cartoonizer Tutorial

In this guide you'll learn how to build, deploy and share your own interactive image-to-image Generative AI web-application using OctoAI!


[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tmoreau89-cartoonize-cartoonizer-gwpv6p.streamlit.app/)

# Requirements

Let's take a look at the requirements in terms of skill set, hardware and software for each phase of this tutorial.

## Phase 1: Experiment with Image to Image feature in Stable Diffusion Web UI

### Time
* 30-60mins depending on how much you want to experiment with Stable Diffusion.

### Skillset
* Git, and beginer-level command line programming.

### Hardware
* A computer of instance with a beefy GPU (Nvidia A10 or better if you want to run the CLIP interrogator model to perform image to text).

### Software
* [AUTOMATIC1111 stable diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) which is under AGPL-3.0 license.
* A model checkpoint of your choice from [CivitAI's website](https://civitai.com/).
* An image of your choice.


## Phase 2: Build your own CLIP-Interrogator Docker container

### Time
* 15-30 mins.

### Skillset
* Git and beginner-level command line and Python programming.

### Hardware
* A computer of instance with a beefy GPU (Nvidia A10 or better if you want to run the CLIP interrogator model to perform image to text).

### Software
* [Docker Engine](https://docs.docker.com/engine/) and a [Docker Hub](https://hub.docker.com/) account.
* [pharmapscychotic clip interrogator library](https://github.com/pharmapsychotic/clip-interrogator) which available under MIT license.
* The ["Build a Container from Python" guide](https://docs.octoai.cloud/docs/create-custom-endpoints-from-python-code) from OctoAI's website.

## Phase 3: Deploy your inference REST endpoint

### Time
* 5-15 mins.

### Skillset
* No experience needed to launch the OctoAI endpoint. Beginner-level command line programming for testing.

### Hardware
* Your laptop.

### Software
* [OctoAI's compute service](https://docs.octoai.cloud/docs) with a user account.
* The CLIP-Interrogator model container built in Phase 2 and uploaded on [Docker Hub](https://hub.docker.com/).
* A Stable Diffusion model container that we built for you.
* A test script to exercise your newly-launched OctoAI endpoint.s

## Phase 4: Build your own Streamlit Web Frontend

### Time
* 15-30 mins.

### Skillset
* No web-design experience needed! Beginner-level Python programming.

### Hardware
* Your laptop.

### Software
* [Streamlit](https://streamlit.io/) library that you can pip-install.
* The OctoAI CLIP-Interrogator inference endpoint URL that you launched in Phase 3.
