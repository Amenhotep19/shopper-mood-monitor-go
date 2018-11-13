package main

import (
	"flag"
	"fmt"
	"os"

	"gocv.io/x/gocv"
)

var (
	// deviceID is camera device ID
	deviceID int
	// input is path to image or video file
	input string
	// faceModel is path to .bin file of model containing face recognizer
	faceModel string
	// faceConfig is path to .xml file of model containing model configuration
	faceConfig string
	// faceConfidence is confidence factor for face detection
	faceConfidence float64
	// sentModel is path to .bin file of sentiment model
	sentModel string
	// sentConfig is path to .xml file of sentiment model containing model configuration
	sentConfig string
	// sentConfidence is confidence factor for emotion detection required
	sentConfidence float64
	// backend is inference backend
	backend int
	// target is inference target
	target int
	// mqttRate is number of seconds between data updates to MQTT server
	mqttRate int
)

func init() {
	flag.IntVar(&deviceID, "device", 0, "Camera device ID")
	flag.StringVar(&input, "input", "", "Path to image or video file")
	flag.StringVar(&faceModel, "face-model", "", "Path to .bin file of face detection model")
	flag.StringVar(&faceConfig, "face-config", "", "Path to .xml file of face detection model configuration")
	flag.Float64Var(&faceConfidence, "face-confidence", 0.5, "Confidence factor for face detection")
	flag.StringVar(&sentModel, "sent-model", "", "Path to .bin file of sentiment detection model")
	flag.StringVar(&sentConfig, "sent-config", "", "Path to .xml file of sentiment detection model configuration")
	flag.Float64Var(&sentConfidence, "sent-confidence", 0.5, "Confidence factor for sentiment detection")
	flag.IntVar(&backend, "backend", 0, "Inference backend. 0: Auto, 1: Halide language, 2: Intel DL Inference Engine")
	flag.IntVar(&target, "target", 0, "Target device. 0: CPU, 1: OpenCL, 2: OpenCL half precision, 3: VPU")
	flag.IntVar(&mqttRate, "mqtt-rate", 1, "Number of seconds between data updates to MQTT server")
}

func parseCliFlags() error {
	// parse cli flags
	flag.Parse()

	// path to face detection model can't be empty
	if faceModel == "" {
		return fmt.Errorf("invalid path to .bin file of face detection model: %s", faceModel)
	}
	// path to face detection model config can't be empty
	if faceConfig == "" {
		return fmt.Errorf("invalid path to .xml file of face detection model configuration: %s", faceConfig)
	}
	// path to sentiment detection model can't be empty
	if sentModel == "" {
		return fmt.Errorf("invalid path to .bin file of sentiment detection model: %s", sentModel)
	}
	// path to sentiment detection model config can't be empty
	if sentConfig == "" {
		return fmt.Errorf("invalid path to .xml file of sentiment detection model configuration: %s", sentConfig)
	}

	return nil
}

func main() {
	// parse cli flags
	if err := parseCliFlags(); err != nil {
		fmt.Fprintf(os.Stderr, "\nERROR: %s\n", err)
		os.Exit(1)
	}

	// open camera device
	cam, err := gocv.VideoCaptureDevice(deviceID)
	if err != nil {
		fmt.Printf("error opening video capture device: %v\n", deviceID)
		return
	}
	defer cam.Close()

	// open display window
	window := gocv.NewWindow("OpenVINO window")
	defer window.Close()

	// prepare input image matrix
	img := gocv.NewMat()
	defer img.Close()

	fmt.Printf("start reading camera device: %v\n", deviceID)
	for {
		if ok := cam.Read(&img); !ok {
			fmt.Printf("cannot read device %v\n", deviceID)
			return
		}
		if img.Empty() {
			continue
		}

		// show the image in the window, and wait 1 millisecond
		window.IMShow(img)
		window.WaitKey(1)
	}
}
