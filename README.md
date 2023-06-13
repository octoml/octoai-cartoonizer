# OctoAI: Cartoonizer Tutorial :camera::star2::octopus:

In this guide you'll learn how to build, deploy and share your own interactive and engaging image-to-image Generative AI web-application using OctoAI!


[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tmoreau89-cartoonize-cartoonizer-gwpv6p.streamlit.app/)

[Video Link](https://drive.google.com/file/d/1706jz3K1BxmvZ5lbB0i_maPZmh89aDMS/view?usp=drive_link)

## Requirements :clipboard:

Let's take a look at the requirements in terms of skill set, hardware and software for each phase of this tutorial.

In total this tutorial takes 1-2 hours depending on how much programming experience you have and how much you want to experiment with stable diffusion functionality. Note that we've designed this tutorial to be as approachable as possible. Very minimal programming experience is required here to be successful.

### Phase 1: Experiment with Image to Image feature in Stable Diffusion Web UI :woman::arrow_right::princess:

#### Time :clock1:
* 30-60mins depending on how much you want to experiment with Stable Diffusion.

#### Skillset :hatching_chick:
* Git, and beginer-level command line programming.

#### Hardware :computer:
* A computer of instance with a beefy GPU (Nvidia A10 or better).

#### Software :floppy_disk:
* [AUTOMATIC1111 stable diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) which is under AGPL-3.0 license.
* A model checkpoint of your choice from [CivitAI's website](https://civitai.com/).
* An image of your choice.


### Phase 2: Build your own CLIP-Interrogator Docker container :package:

#### Time :clock1230:
* 15-30 mins.

#### Skillset :hatching_chick:
* Git and beginner-level command line and Python programming.

#### Hardware :computer:
* A computer of instance with a decent GPU (Nvidia T4 or better).

#### Software :floppy_disk:
* [Docker Engine](https://docs.docker.com/engine/) and a [Docker Hub](https://hub.docker.com/) account.
* [pharmapscychotic clip interrogator library](https://github.com/pharmapsychotic/clip-interrogator) which available under MIT license.
* The ["Build a Container from Python" guide](https://docs.octoai.cloud/docs/create-custom-endpoints-from-python-code) from OctoAI's website.

### Phase 3: Deploy your inference REST endpoint :octopus:

#### Time :clock1230:
* 5-15 mins.

#### Skillset :hatching_chick:
* No experience needed to launch the OctoAI endpoint. Beginner-level command line programming for testing.

#### Hardware :computer:
* Your laptop.

#### Software :floppy_disk:
* [OctoAI's compute service](https://docs.octoai.cloud/docs) with a user account.
* The CLIP-Interrogator model container built in Phase 2 and uploaded on [Docker Hub](https://hub.docker.com/).
* A Stable Diffusion model container that we built for you.
* A test script to exercise your newly-launched OctoAI endpoint.s

### Phase 4: Build your own Streamlit Web Frontend :technologist:

#### Time :clock1230:
* 15-30 mins.

#### Skillset :hatching_chick:
* No web-design experience needed! Beginner-level Python programming.

#### Hardware :computer:
* Your laptop for developing your webapp, a laptop or phone/tablet to test the webapp.

#### Software :floppy_disk:
* [Streamlit](https://streamlit.io/) library that you can pip-install.
* The OctoAI inference endpoint URLs that you launched in Phase 3.

## Disclaimer

Under construction.

## Step-by-step Cartoonizer Tutorial :books:

### Phase 1: Experiment with Image to Image feature in Stable Diffusion Web UI :woman::arrow_right::princess:

There is a huge community of stable diffusion enthusiasts out there so whether you find your information on Reddit, youtube, or GitHub, you're sure to find quality content.

I learned how to use Stable diffusion's image to image feature on Youtube actually thanks to the following [video](https://www.youtube.com/watch?v=dSn_vXrjaK8).

You'll find that the community of stable diffusion enthusiasts gravitate around [AUTOMATIC1111's Stable Diffusion Web UI tool](https://github.com/AUTOMATIC1111/stable-diffusion-webui). You'll find that it's a pretty easy tool to onboard with. Follow the installation instructions available [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui#installation-and-running).

I highly recommend a machine that has a pretty capable GPU, on AWS I recommend using an instance that has an A10, e.g. a `g5.4xlarge`.

Once you've installed the dependencies, and have cloned the repo locally on your development machine, you can start downloading your favorite checkpoints from CivitAI's website. Download the checkpoints under `stable-diffusion-webui/models/Stable-diffusion/`, and make sure that the files have a `.safetensors` extension.

You can launch the WebUI using the following command:
```bash
./webui.sh --share --xformers
```

Look at what the script prints out - particularly be on the lookout for (the URL will decidedly be different than the one below):
```
Running on public URL: https://80870b04f9f578c8cf.gradio.live
```

This is the URL that you'll paste in your browser to start playing with the Web UI! Once the Web UI has launched, click on the img2img tab to start playing with the image to image generation functionality.

![stable-diffusion-web-ui](assets/stable-diffusion-web-ui.png)


Select under `Stable Diffusion checkpoint` the model file that you downloaded under `stable-diffusion-webui/models/Stable-diffusion/`. This will determine the style you get on the output image.

Start by uploading a photo of your choice. In order for image to image to work well, you'll need to provide a textual description of the image you upload under the prompt. This is because Stable Diffusion requires test in order to guide the image that it generates. It would be very cumbersome to have to describe each picture we upload manually, so instead we can use the `Interrogate CLIP` button which invokes a CLIP Interrogator model. This essentially performs the reverse of Stable Diffusion's text-to-image: from an image, it gives you text.

Once you've interrogated the CLIP model, you will be able to see what it has inferred from the image you uploaded. Now you can hit the "Generate" button to see what image it has generated.

It should look pretty convincing overall. But you can spend a lot of time in the GUI to tweak several settings. Starting with `Denoising strength` which you can scale down or up to see how it impacts the generated image. Give it a try!

If you're tired hitting generate button after setting a knob to a different value, you can use the handy `X/Y/Z plot` under the `Script` pull down menu. This lets you sweep across different parameters easily to study the effect these parameters have on your generated image. 

For instance you can set `X type` to `Denoising` and under `X values` sweep values from 0.1 to 0.9 in +0.1 increments by entering `0.1-0.9(+0.1)`. Hit generate and see what you get! You can explore up to 3 parameters in this fashion, really handy!

![cgi_sweep_experiment](assets/cgi_sweep_experiment.png)

Hopefully by playing around long enough with the Stable Diffusion Web UI, you can build really good intuition on how Stable Diffusion works, and build some very neat media. We're going to use that knowledge back into our web app to design a simple, single-shot "cartoonizer" app.

### Phase 2: Build your own CLIP-Interrogator Docker container :package:

Let's walk through how you can build your own model container and upload it on DockerHub. Once uploaded you can use OctoAI's compute service to serve the model container behind an easy to use REST endpoint. This makes your model easy to access, manage and scale based on your use requirements.

#### A. Test out CLIP-Interrogator

As we saw in Phase 1, image to image generation depends on being able to generate a prompt from an input image. This promp will be used to guide the generation of the output "cartoonized" image.

For this exercise, we'll use [pharmapscychotic clip interrogator library](https://github.com/pharmapsychotic/clip-interrogator) which available under MIT license.

We follow the instructions available to their website. Namely from a computer/instance that you can program on, open a terminal (or ssh into it) and do the following.

Create and activate a python virtual environment. This will create a new local environment where you can install pip packages without needing to worry about breaking dependencies for your other Python projects.

```bash
cd $MY_CLIP_TEST_DIRECTORY
python3 -m venv .venv
source .venv/bin/activate
```

Now install with pip.
```bash
# install torch with GPU support for example:
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117

# install clip-interrogator and Pillow (image library)
pip3 install clip-interrogator==0.5.4
pip3 install Pillow
```

You are ready to test out CLIP on an image of your choice with the following python test script that you can save in `test_clip.py`. Replace `my_test_image.png` below to point to your input image.
```python
# Import Pillow image library
from PIL import Image
# Import CLIP Interrogator library
from clip_interrogator import Config, Interrogator

# Open image
image = Image.open(my_test_image.png).convert('RGB')
# Initialize interrogator model with OpenAI's ViT-L-14 sentence transformer (https://huggingface.co/sentence-transformers/clip-ViT-L-14)
ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))

# Invoke the interrogator, and print the text that it returns
print(ci.interrogate(image))
```

#### B. Familiarize yourself with OctoAI's guide to building a model container

We're going to use the [Build a Container from Python Guide](https://docs.octoai.cloud/docs/create-custom-endpoints-from-python-code) as a guide to build our own Clip-Interrogator model. Don't be intimidated by the fact that this guide is labeled as "advanced".

Let's start with the files. All you need are the following files to build your model container:
* `model.py` is where we define how to run an inference on your model of choice
* `server.py` wraps the mode in a [Sanic](https://sanic.dev/en/) server
* `requirements.txt` that lists all of the python libraries that the model container requires to run properly
* `Dockerfile` which installs all of the dependencies, and downloads the model files into a Docker image at build time so that at deployment time, our model container is ready to fire!

#### C. Build your own model container - a step by step guide

To help you, we've provided all of the code you need to build your own CLIP-Interrogator model container under [model_containers/clip_interrogator](model_containers/clip_interrogator). But let's go over how we managed to write this code so you can replicate this for models of your choice!

##### i. `model.py`
Let's edit the `model.py` file from the guide. The guide assumes that we're packaging a FLAN-T5-small text generation model into a model container. We need to make a few edits to this file to replace that model with our CLIP-Interrogator model.

Let's first replace:
```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
```

with:
```python
from base64 import b64decode
from io import BytesIO
from PIL import Image
from clip_interrogator import Config, Interrogator
```

We don't need the `transformers` library in the model file. What we'll need are libraries to process images from a byte input format (`base64`, `io`, `PIL`) because that's how it will be sent over to the model inference server. Then we'll need the `clip_interrogator` library to initialize and invoke our image to text model.

Next we'll need to modify the `Model` class definition.

Let's replace the `__init__()` method:

```python
    def __init__(self):
        """Initialize the model."""
        self._tokenizer = T5Tokenizer.from_pretrained(_MODEL_NAME)
        self._model = T5ForConditionalGeneration.from_pretrained(_MODEL_NAME).to(
            _DEVICE
        )
```

with:
```python
    def __init__(self):
        """Initialize the model."""
        self._clip_interrogator = Interrogator(Config(
                clip_model_name="ViT-L-14/openai",
                clip_model_path='cache',
                device="cuda:0" if torch.cuda.is_available() else "cpu"))
```
Here the idea is to initialize the model as we did in our experiments in part A. This time around we'll specify to use the GPU as the preferred device to run the model on. 

Finally, let's replace the `predict()` method:

```python
    def predict(self, inputs: typing.Dict[str, str]) -> typing.Dict[str, str]:
        """Return a dict containing the completion of the given prompt.

        :param inputs: dict of inputs containing a prompt and optionally the max length
            of the completion to generate.
        :return: a dict containing the generated completion.
        """
        prompt = inputs.get("prompt", None)
        max_length = inputs.get("max_length", 2048)

        input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids.to(_DEVICE)
        output = self._model.generate(input_ids, max_length=max_length)
        result = self._tokenizer.decode(output[0], skip_special_tokens=True)

        return {"completion": result}
```

with:

```python
    def predict(self, inputs: typing.Dict[str, typing.Any]) -> typing.Dict[str, str]:
        """Return interrogation for the given image.

        :param inputs: dict of inputs containing model inputs
               with the following keys:

        - "image" (mandatory): A base64-encoded image.
        - "mode" (mandatory): A

        :return: a dict containing these keys:

        - "labels": String containing the labels.
        """

        image = inputs.get("image", None)
        mode = inputs.get("mode", "default")
        image_bytes = BytesIO(b64decode(image))
        image_dat = Image.open(image_bytes).convert('RGB')
        if mode == "fast":
            outputs = self._clip_interrogator.interrogate_fast(image_dat)
        elif mode == "classic":
            outputs = self._clip_interrogator.interrogate_classic(image_dat)
        elif mode == "negative":
            outputs = self._clip_interrogator.interrogate_negative(image_dat)
        else:
            outputs = self._clip_interrogator.interrogate(image_dat)

        response = {"completion": {"labels": outputs}}

        return response
```

This is really the hardest part of this tutorial (but as you see it's easy to follow). The idea here is that `predict()` takes in an input dictionary, which contains an `image: base-64 encoded image` and a `mode: string` key-value pair. The mode string can be one of `fast`, `classic`, `negative` and `default` which provide different [interrogation modes](https://github.com/pharmapsychotic/clip-interrogator/blob/main/clip_interrogator/clip_interrogator.py#L205-L255) offered by the CLIP interrogator. To quote the code comments:

| Mode | Explanation|
|------|------------|
| Fast mode | Simply adds the top ranked terms after a caption. It generally results in better similarity between generated prompt and image than classic mode, but the prompts are less readable. |
| Classic mode | Creates a prompt in a standard format first describing the image, then listing the artist, trending, movement, and flavor text modifiers. |
| Negative mode | Chains together the most dissimilar terms to the image. It can be used to help build a negative prompt to pair with the regular positive prompt and often improve the results of generated images particularly with Stable Diffusion 2. |
| Default mode | Default interrogation mode. |

It returns an output dictionary that contains an `labels: label_string` key-value pair.

That's a wrap for what we need to change in `model.py`!

##### ii. `server.py`

The beauty is - you don't have to change a line here - server code implementation stays pretty constant across different model container implementations.

##### iii. `requirements.txt`

Remove the following requirements which we won't need:
```
transformers==4.27.4
sentencepiece==0.1.97
```

and replace it with, which echo the packages that we installed in Section A. The transformers library has to be set to a slighly older version to work with the `clip-interrogator` library.
```
transformers==4.26.0
Pillow>=6.2.1
clip-interrogator==0.5.4
```

##### iv. `Dockerfile`

Again, no need to make changes here!

##### v. `Makefile`

The [Create Custom Endpoints from Python Code](https://docs.octoai.cloud/docs/create-custom-endpoints-from-python-code) guide does not include a makefile, but I've included one that's just going to make our lives a bit easier.

```makefile
IMAGE_TAG ?= clip-interrogator
CONTAINER_NAME ?= ${IMAGE_TAG:%:latest=%}
INFERENCE_SERVER_PORT ?= 8000

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

build:
	docker build -t "${IMAGE_TAG}" --progress string .

run:
	docker run -d --rm -p ${INFERENCE_SERVER_PORT}:8000 \
		--gpus all \
		--name "${CONTAINER_NAME}" "${IMAGE_TAG}"

stop:
	docker stop "${CONTAINER_NAME}"

.PHONY: build run stop
```

Thanks to this `Makefile` you can use make commands to accomplish the following.

| Make command | Task |
|--------------|------|
| make build | build the `clip-interrogator` container (takes a few minutes) |
| make run | runs the `clip-interrogator` container so you can test it |
| make stop | stops the `clip-interrogator` container when you are done testing it |

In your model container directory, all you have to do is run the following commands.

```bash
cd $MY_CLIP_INTERROGATOR_MODEL_CONTAINER_DIRECTORY
make build # takes a couple of minutes
make run # runs docker container

docker container ls # shows you what containers are running - you should see the clip-interrogator image
docker logs clip-interrogator # dumps the logs from that container
```

From dumping the logs, you'll see the following after ~30s goes by.

```text
Loading BLIP model...
load checkpoint from https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth
Loading CLIP model...
Loaded CLIP model and data in 5.63 seconds.
```

Your container is running locally (i.e. localhost) and listening in on port 8000. Time to test it out!

#### D. Testing your model container

You now have a model container server running and listening in for client requests. Let's build a simple script that will exercise the CLIP-Interrogator model container. Under `test_curl.sh` write the following bash code. Note that you'll have to change `my_test_image.png` to whatever photo you want to use to test the CLIP-Interrogator model.

```bash
#!/bin/bash

set -euxo pipefail

docker_port=$(docker port clip-interrogator 8000/tcp)
ENDPOINT=localhost:${docker_port##*:}

echo "{\"mode\": \"fast\", " > request.json
echo "\"image\": \"" >> request.json
base64 my_test_image.png >> request.json
echo "\"}" >> request.json

curl -X POST http://${ENDPOINT}/predict \
    -H "Content-Type: application/json" \
    --data @request.json > response.json
```

Locally, you'll see that new files have popped up. One named `request.json` contains the REST request data sent to the endpoint. The one named `response.json` contains the REST response received from the endpoint.

Execute the test with the following
```bash
chmod u+x test_curl.sh
./test_curl.sh
```

Because we're using the `fast` clip interrogator mode in this test, the response should take about 2 seconds to come back. Open the `response.json` file in a text editor and look at is contents - you should find in its contents a description of the `my_test_image.png` image you picked -- for instance:

```
{"completion":{"labels":"a close up of a person wearing glasses and a scarf, ..., wearing small round glasses"}}
```

To see what happened on the server side, remember you can always look at the logs with `docker logs clip-interrogator`.

This time, try out the other modes (`classic`, `default`, `negative`) and see how it impacts the labels you get and also the time it takes to generate the labels!

When you're finished running the tests, don't forget to stop the container with the following make command: 
```bash
make stop
```

#### E. Uploading your model to a Docker Hub repository

We've build our model container and have tested it. It's ready to be uploaded to [Docker Hub](https://hub.docker.com/)!

First create an account on [their website](https://hub.docker.com/) if you haven't done so already. On your dashboard, you can now create a new repository, by clicking on the `Create repository button`.

Under your namespace, let's name our repository `clip-interrogator`, and describe it as you see fit. Set the Visibility to `Public` instead of `Private`. Hit the `Create` button.

Once that's been created, we're ready to upload our newly minted model container to this repository. Back to the terminal run the following and replace $DOCKER_USERNAME with your actual docker hub username:

```bash
docker login # enter your dockerhub credentials
docker tag clip-interrogator:latest $DOCKER_USERNAME/clip-interrogator:v0.1.0 # tag your image and version it
docker push $DOCKER_USERNAME/clip-interrogator:v0.1.0 # push it!
```

It can take a few minutes to push the docker image to the repository depending on the speed of your internet connection. Note that this image is about 7GB big because we've downloaded the models weights into it, it'll affects how long it takes to upload it.

Once the upload has completed, go on your docker hub landing page to see if the new image has appeared under the repository you have just created.

![dockerhub](assets/dockerhub.png)

Success! We're now ready to launch our nifty CLIP-Interrogator inference endpoint.

### Phase 3: Deploy your inference REST endpoints :octopus:

In this part of the tutorial we're going to deploy two OctoAI compute service inference endpoints.

One to serve the CLIP-Interrogator model container that we just built and uploaded into Docker Hub. Another one to serve the [Stable Diffusion model container](https://hub.docker.com/repository/docker/tmoreau89octo/cartoonizer-stable-diffusion/general) that we've built for your convenience. You are of course free or even encouraged to build your own based on what you learned in Phase 2 of this tutorial!

#### A. Launching a CLIP-Interrogator endpoint

Go to the [OctoAI compute service website](https://octoai.cloud/). Hit `Login` at the bottom left of the page and log in with your preferred authentication method (email, Google or GitHub).

Time to launch our endpoint. If you want to have a detailed guide on how to do this, check out the [Create a Custom Endpoint from a Container](https://docs.octoai.cloud/docs/create-custom-endpoints-from-a-container) guide. If you don't want to read, that's okay, it should be really straightforward!

So let's click on `Endpoints` at the top left of the OctoAI landing page. Next, click on `New endpoint` to create your very first endpoint!
* Under `Endpoint name`, enter "clip-interrogator"
* Under `Container Image`, enter the tag you used to upload your model to dockerhub. You can always use the one we build for the tutorial: ["tmoreau89octo/cartoonizer-clip-interrogator:v0.1.0"](https://hub.docker.com/layers/tmoreau89octo/cartoonizer-clip-interrogator/v0.1.0/images/sha256-f9b5e650856d1020013fcebae02d65dfba4bb2825d5507d37d85e7ccd3f5dc35?context=repo).
* Under `Container Port`, we're going to use port "8000".
* Keep the `Registry Credential` unchanged ("Public").
* Keep the `Health check path` unchanged ("/healthcheck").
* Turn on the `Enable public access` toggle switch.
* No need to specify secrets.
* Under `Select Hardware` click on the "Small 16GB: instance that is equipped with an T4 GPU.
* Set `Min Replicas` to "1", and `Max Replicas` to "1" for this experiment.
* Leave the timeout as is.

![octoai](assets/octoai.png)

Now hit `Create`! You'll get informed that the very first inference will undergo several minutes of cold start. This is normal. Wait for the blue square under `Replicas` to turn solid - it usually takes a minute or two. That's how you know that the endpoint is up and ready to be tested! You can check out the logs by clicking on the `Logs` button at the top right. It should look very much like the logs of the docker container you ran locally in Phase 3, Section C.

Note the Endpoint URL! We're going to use this one in a second to test our endpoint. To test your brand new OctoAI endpoint, we cna re-use the `test_curl.sh` script used in Phase 3, Section D with a small modification. Change the following line:

```bash
curl -X POST http://${ENDPOINT}/predict \
```

with the Endpoint URL of your newly launched OctoAI compute server with `/predict` appended to it:
```bash
curl -X POST https://clip-interrogator-4jkxk521l3v1.octoai.cloud/predict \
```

As a sanity check, check the inference server logs to see if your model container has received the request. And finally check the `response.json` file to find a lable that should match the one that was returned by the model container we were testing locally in Phase 3, Section D. Amazing!

#### B. Launching a Stable-Diffusion endpoint

Repeat the same steps here but this time around we'll give you the URL to the model container you'll be launching with OctoAI. Keep all of the parameters the same. 
* Under `Endpoint name`, enter "stable-diffusion"
* Under Under `Container Image`, point to the following `tmoreau89octo/cartoonizer-stable-diffusion:v0.1.0` container, available under this [link](https://hub.docker.com/layers/tmoreau89octo/cartoonizer-stable-diffusion/v0.1.0/images/sha256-cdf204d7ceb81ba39d92e8c3502a272de249510c442ccd142a15147395a13cae?context=repo).

Perform the same steps. We've already tested this one for you, so you'll exercise it when you build the full cartoonizer web app.

Got your two endpoints up and running? Let's go and build the web app.

#### C. Stopping your endpoints

Just like when you use cloud instances, it's suboptimal to let them run indefinitely (particularly if your min-replicas is set to 0). So when you're done with this tutorial, you can stop you OctoAI endpoint by clicking on the `Endpoint` button on the top left of the OctoAI landing page, select the endpoint that you want to stop, hit `Edit` at the top left, and hit `Delete`.

### Phase 4: Build your own Streamlit Web Apps

Yes, we have our AI endpoints up! Time to build our web app! No web design experience? No problem. All you need is a bit of Python experience, thanks to the Streamlit library that lets you quickly build simple web apps and host them for free as long as you host your source code on a public GitHub repo. You'll need to create a [Streamlit](https://streamlit.io/) account.

We're going to build our web app in two stages. First we're going to test an interactive CLIP-Interrogator web app that lets us upload an image and see how the AI labels it. Second, we'll build our Cartoonizer web app that will turn images that you upload into a cartoon version of yourself.

#### A. A simple interactive CLIP-Interrogator frontend :technologist:

You can get all of the code you need to have that fronend up and running under [websites/clip_interrogator/](websites/clip_interrogator/). But to build it from scratch, follow the instructions below:

On your laptop or preferred development machine, set up your python environment so we can install Streamlit.

```bash
cd $MY_CLIP_TEST_DIRECTORY
python3 -m venv .venv
source .venv/bin/activate
```

Now install the pip packages you need to test your CLIP-Interrogator web app.

```bash
pip3 install streamlit
pip3 install Pillow
```

Let's create a Python file that implements our interactive web app using Streamlit, and we'll call it `clip_interrogator.py`. In under 50 lines of code, you can build a web app that lets you set the mode of the interrogation of the CLIP-Interrogator model, upload an image, process it to be sent to the CLIP-Interrogator OctoAI inference endpoint, and display the label that endpoint returns! Note that some modes will be significantly slower to run than others. If you want fast processing times, I suggest you stick to `fast` mode.

```python
import streamlit as st
from PIL import Image
from io import BytesIO
from base64 import b64encode
import requests

def run_clip_interrogator(upload, mode):
    # Input image that the user is uploading
    input_img = Image.open(upload)
    # Apply cropping and resizing to work on a square image
    st.write("Input Image :camera:")
    st.image(input_img)
    # Prepare the JSON query to send to OctoAI's inference endpoint
    buffer = BytesIO()
    input_img.save(buffer, format="png")
    image_out_bytes = buffer.getvalue()
    image_out_b64 = b64encode(image_out_bytes)
    model_request = {
        "image": image_out_b64.decode("utf8"),
        "mode": mode
    }
    # Send the model request!
    reply = requests.post(
        f"https://clip-interrogator-4jkxk521l3v1.octoai.cloud/predict",
        headers={"Content-Type": "application/json"},
        json=model_request
    )
    # Print the labels returned by the inference endpoint
    labels =reply.json()["completion"]["labels"]
    st.write("Labels from CLIP-Interrogator: {}".format(labels))

# The Webpage
st.set_page_config(layout="wide", page_title="CLIP Interrogator")
st.write("## CLIP Interrogator - Powered by OctoAI")
st.markdown(
    "Select a mode, upload a photo and let the CLIP Interrogator model determine what it sees in it!"
)
# User-controllable radio dials to select CLIP interrogator mode
mode = st.radio(
        'CLIP mode',
        ("default", "classic", "fast", "negative"))
# Upload button to upload your photo
my_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    run_clip_interrogator(my_upload, mode)
```

Now that you've saved your Python file, you can test it on your local browser by just running the following command in the directory that contains the `clip-interrogator.py` file.

```bash
streamlit run clip_interrogator.py
```

Assuming you're not SSH-ed into the machine you run this from, Streamlit will automatically run your page in the browser. If you're SSHing into the machine Streamlit will give you a Local and Network URL you can access from your browser assuming the machine firewall allows external connections through the ports used by Streamlit.

Select your CLIP Interrogator mode, upload a picture of your choice, and feel the power of offloading all of that heavy AI compute to the OctoAI compute service. Neat!

![clip-interrogator-web-app](assets/clip-interrogator-web-app.png)

Next we'll build the full Cartoonizer app by invoking both CLIP-Interrogator and Stable-Diffusion OctoAI endpoints from within your web app.

Stay tuned - WIP...
