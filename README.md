# Shopper Mood Monitor

| Details            |              |
|-----------------------|---------------|
| Target OS:            |  Ubuntu\* 16.04 LTS   |
| Programming Language: |  Go* |
| Time to Complete:    |  45 min     |

![app image](./images/shopper-mood-monitor.png)

## Introduction

This shopper mood monitor application is one of a series of reference implementations for Computer Vision (CV) using the Intel® Distribution of OpenVINO™ toolkit written in the Go* programming language. This application is designed for a retail shelf mounted camera system that counts the the number of passers-by that look toward the display. Their mood is captured, and is classified into 5 emotions:

- Neutral
- Happy
- Sad
- Surprised
- Anger

If the emotion can't be determined with a level of confidence higher than the value configurable via command line parameter, the emotion is marked as Unknown.

It is intended to provide real-world marketing statistics for in-store shopping mood analysis.

## Requirements

### Hardware

* 6th Generation Intel® Core™ processor with Iris® Pro graphics and Intel® HD Graphics

### Software

* [Ubuntu\* 16.04 LTS](http://releases.ubuntu.com/16.04/)
*Note*: You must be running kernel version 4.7+ to use this software. We recommend using a 4.14+ kernel to use this software. Run the following command to determine your kernel version:

```shell
uname -a
```

* OpenCL™ Runtime Package
* Intel® Distribution of OpenVINO™ toolkit
* Go programming language v1.11+

## Setup

### Install Intel® Distribution of OpenVINO™ toolkit

Refer to https://software.intel.com/en-us/articles/OpenVINO-Install-Linux for more information about how to install and setup the Intel® Distribution of OpenVINO™ toolkit.

You will need the OpenCL™ Runtime package if you plan to run inference on the GPU as shown by the instructions below. It is not mandatory for CPU inference.

### Install Go*

You must install the Go programming language version 1.11+ in order to compile this application. You can obtain the latest compiler from the Go website's download page at https://golang.org/dl/

For an excellent introduction to the Go programming language, check out the online tour at https://tour.golang.org

### Download the reference platform code using "go get"

You can download the reference platform code onto your computer by using the following Go command:

```shell
go get -d github.com/intel-iot-devkit/shopper-mood-monitor-go
```

Then, change the current directory to where you have installed the application code to continue the installation steps:

```shell
cd $GOPATH/src/github.com/intel-iot-devkit/shopper-mood-monitor-go
```

### Install Dep

This sample uses the `dep` dependency tool for Go. You can download and install it by running the following command:

```shell
make godep

```

### Install GoCV

Once you have installed Go, you must also install the GoCV (https://gocv.io/) package which contains the Go programming language wrappers for the Intel® Distribution of OpenVINO™ toolkit, and the associated dependencies. The easiest way to do this is by using the `dep` tool, which will satisfy the program's dependencies as defined in `Gopkg.lock` file. Run the following make file task to do so:

```shell
make dep
```

Now you should be ready to build and run the reference platform application code.

## How it Works

The application uses a video source, such as a camera, to grab frames, and then uses 2 different Deep Neural Networks (DNNs) to process the data. The first network looks for faces, and then if successful, counts the face as a "shopper"

A second neural network is then used to determine the emotion for each detected face, if the person's head is facing towards the camera.

The data can then optionally be sent to a MQTT machine to machine messaging server, as part of a retail data analytics system.

The DNN models used are Intel® optimized models that are part of the Intel® Distribution of OpenVINO™ toolkit.

You can find them here:

- `/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001`
- `/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003`

![Code organization](./images/arch3.png)

The program creates three go routines for concurrency:

- Main goroutine that performs the video i/o
- Goroutine that processes video frames using the deep neural networks
- Goroutine that publishes any MQTT messages

## Setting the Build Environment

You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:

```shell
source /opt/intel/computer_vision_sdk/bin/setupvars.sh
```

## Building the Code

Start by changing the current directory to wherever you have git cloned the application code. For example:

```shell
cd $GOPATH/src/github.com/intel-iot-devkit/shopper-mood-monitor-go
```

Once you have fetched the dependencies you must export a few environment variables required to build the library from the fetched dependencies. Run the following command from the project directory:

```shell
source vendor/gocv.io/x/gocv/openvino/env.sh
```

Now you are ready to build the program binary. The project ships a simple `Makefile` which makes building the program easy by invoking the `build` task from the project root as follows:

```shell
make build
```

This commands creates a new directory called `build` in your current working directory and places the newly built binary called `monitor` into it.

## Running the Code

To see a list of the various options:

```shell
./build/monitor -h
```

To run the application with the needed models using the webcam:

```shell
./build/monitor -face-model=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.bin -face-config=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml -sent-model=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.bin -sent-config=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml
```

The user can choose different confidence levels for both face and emotion detection by using `-face-confidence` and `-sent-confidence`. command line parameters. By default, both of these parameters are set to `0.5` i.e. at least `50%` confidence is required in order for the returned inference result to be considered valid.

### Hardware Acceleration

This application can take advantage of the hardware acceleration in the Intel® Distribution of OpenVINO™ toolkit by using the `-backend, -b` and `-target, -t` parameters.

For example, to use the Intel® Distribution of OpenVINO™ toolkit backend with the GPU in 32-bit mode you need to set the `-backend` flag to `2` and `-target` flag to `1`:

```shell
./build/monitor -face-model=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.bin -face-config=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml -sent-model=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.bin -sent-config=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml -backend=2 -target=1
```

To run the code using 16-bit floats, you have to both set the `-target` flag to use the GPU in 16-bit mode, as well as use the FP16 version of the Intel® models:

```shell
./build/monitor -face-model=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP16/face-detection-adas-0001.bin -face-config=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml -sent-model=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.bin -sent-config=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.xml -backend=2 -target=2
```

To run the code using the VPU, you have to set the `-target` flag to `3` and also use the 16-bit FP16 version of the Intel® models:

```shell
./build/monitor -face-model=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP16/face-detection-adas-0001.bin -face-config=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml -sent-model=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.bin -sent-config=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.xml -backend=2 -target=3
```

## Sample videos

There are several videos available to use as sample videos to show the capabilities of this application. You can download them by running these commands from the `shopper-mood-monitor-go` directory:

```shell
mkdir resources
cd resources
wget https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking-and-pause.mp4
wget https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4
cd ..
```

To then execute the code using one of these sample videos, run the following commands from the `shopper-mood-monitor-go` directory:

```shell
./build/monitor -face-model=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.bin -face-config=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml -sent-model=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.bin -sent-config=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml -input=./resources/face-demographics-walking-and-pause.mp4
```

## Machine to Machine Messaging with MQTT

If you wish to use a MQTT server to publish data, you should set the following environment variables before running the program:

```shell
export MQTT_SERVER=tcp://localhost:1883
export MQTT_CLIENT_ID=shopper123
```

*Note:* You must always specify the server protocol -- the above example is using `TCP` protocol.

Change the `MQTT_SERVER` to a value that matches the MQTT server you are connecting to.

You should change the `MQTT_CLIENT_ID` to a unique value for each monitoring station, so you can track the data for individual locations. For example:

```shell
export MQTT_CLIENT_ID=hopper123
```

If you want to monitor the MQTT messages sent to your local server, and you have the `mosquitto` client utilities installed, you can run the following command:

```shell
mosquitto_sub -t 'retail/traffic'
```

## Docker*

You can also build a Docker* image and then run the program in a Docker container. First you need to build the image. You can use the `Dockerfile` present in the cloned repository and build the Docker image.

First you must obtain your own unique download URL for the Intel® Distribution of OpenVINO™ toolkit. Follow the registration process if you have not yet done so. In the registration email you have received a link to the Registration Center website download page, shown here:

![OpenVINO download page](./images/openvino-download.png)

First, navigate to the download page using the link you have received. On the download page, use the "Choose Product to Download" selection box and select "Intel® Distribution of OpenVINO™ toolkit for Linux". Next, using the "Choose a Version" selection box, select "2018 R5". The "Choose a Download Option" section should appear. Right click on the button "Full Package" and choose "Copy Link Address". Your clipboard should now contain your unique download URL. Save this URL somewhere safe.

Now you can build your unique Docker image by running the following command, substituting the actual URL you obtained in the previous step:

```shell
docker build -t shopper-mood-go --build-arg OPENVINO_DOWNLOAD_URL=[your unique OpenVINO download URL here] .
```

This will produce a Docker image called `shopper-mood-go` which contains the built binary. Since the built Docker image has an [ENTRYPOINT](https://docs.docker.com/engine/reference/builder/#entrypoint) defined you can run the image as an executable using the following command:

```shell
docker run -it --rm shopper-mood-go -h
```

To run the Docker image on an Ubuntu host machine using an attached camera, run the following commands:

```shell
xhost +local:docker
docker run --device=/dev/video0:/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -it --rm shopper-mood-go -face-model=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.bin -face-config=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml -sent-model=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.bin -sent-config=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml
xhost -local:docker
```

To run the Docker image on an Ubuntu host machine using a file input, run the following commands:

```shell
xhost +local:docker
docker run -v ${PWD}/resources:/resources -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -it --rm shopper-mood-go -face-model=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.bin -face-config=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml -sent-model=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.bin -sent-config=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml -input=/resources/face-demographics-walking-and-pause.mp4
xhost -local:docker
```

## Microsoft Azure*

If you'd like to know how you can take advantage of more advanced build system provided by [Microsoft Azure* Cloud](https://azure.microsoft.com/), please check out the Microsoft Azure* guide [here](./azure.md). Following the steps in the guide you can build a Docker container and push it into Azure Container Registry to make it available online.
