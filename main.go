package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"gocv.io/x/gocv"
)

const (
	// name is a program name
	name = "shopper-mood-monitor"
	// topic is MQTT topic
	mqttTopic = "retail/traffic"
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
	// sentConfidence is confidence factor for sentiment detection
	sentConfidence float64
	// backend is inference backend
	backend int
	// target is inference target
	target int
	// mqttRate is number of seconds between data updates to MQTT server
	mqttRate int
)

// Sentiment is shopper sentiment
type Sentiment int

const (
	// NEUTRAL is neutral emotion shopper
	NEUTRAL Sentiment = iota + 1
	// HAPPY is for detecting happy emotion
	HAPPY
	// SAD is for detecting sad emotion
	SAD
	// SURPRISED is for detecting neutral emotion
	SURPRISED
	// ANGER  is for detecting anger emotion
	ANGER
	// UNKNOWN is catchall unidentifiable emotion
	UNKNOWN
)

// String implements fmt.Stringer interface for Sentiment
func (s Sentiment) String() string {
	switch s {
	case NEUTRAL:
		return "NEUTRAL"
	case HAPPY:
		return "HAPPY"
	case SAD:
		return "SAD"
	case SURPRISED:
		return "SURPRISED"
	case ANGER:
		return "ANGER"
	default:
		return "UNKNOWN"
	}
}

// Result is inference result
type Result struct {
	// Shoppers contains count of all detected faces of shoppers
	Shoppers int
	// Detections is a map which contains detected sentiment of each shopper
	Detections map[Sentiment]int
}

// String implements fmt.Stringer interface for Result
func (r Result) String() string {
	return fmt.Sprintf("Shoppers: %d, Neutral: %d, Happy: %d, Sad: %d, Surprised: %d, Anger: %d, Unknown: %d",
		r.Shoppers, r.Detections[NEUTRAL], r.Detections[HAPPY], r.Detections[SAD],
		r.Detections[SURPRISED], r.Detections[ANGER], r.Detections[UNKNOWN])
}

// ToMQTTMessage turns result into MQTT message which can be published to MQTT broker
func (r Result) ToMQTTMessage() string {
	return fmt.Sprintf("{\"shoppers\": \"%d\", \"neutral\": \"%d\", \"happy\": \"%d\", \"sad\": \"%d\", \"surprised\": \"%d\", \"anger\": \"%d\", \"unknown\": \"%d\"}",
		r.Shoppers, r.Detections[NEUTRAL], r.Detections[HAPPY], r.Detections[SAD],
		r.Detections[SURPRISED], r.Detections[ANGER], r.Detections[UNKNOWN])
}

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
		return fmt.Errorf("Invalid path to .bin file of face detection model: %s", faceModel)
	}
	// path to face detection model config can't be empty
	if faceConfig == "" {
		return fmt.Errorf("Invalid path to .xml file of face detection model configuration: %s", faceConfig)
	}
	// path to sentiment detection model can't be empty
	if sentModel == "" {
		return fmt.Errorf("Invalid path to .bin file of sentiment detection model: %s", sentModel)
	}
	// path to sentiment detection model config can't be empty
	if sentConfig == "" {
		return fmt.Errorf("Invalid path to .xml file of sentiment detection model configuration: %s", sentConfig)
	}

	return nil
}

// messageRunner reads mqtt message from mqttChan with rate frequency and sends them to MQTT server
// doneChan is used to receive a signal from the main goroutine to notify messageRunner to stop and return
func messageRunner(doneChan <-chan struct{}, mqttChan <-chan Result, c *MQTTClient, topic string, rate int) error {
	ticker := time.NewTicker(time.Duration(rate) * time.Second)

	for {
		select {
		case <-ticker.C:
			result := <-mqttChan
			_, err := c.Publish(topic, result.ToMQTTMessage())
			// TODO: decide whether to return with error and stop program;
			// For now we just signal there was an error and carry on
			if err != nil {
				fmt.Printf("Error publishing message to %s: %v", topic, err)
			}
		//case res := <-mqttChan:
		case <-mqttChan:
			// we discard messages in between ticker times
		case <-doneChan:
			fmt.Printf("Stopping messageRunner: received stop sginal\n")
			return nil
		}
	}

	return nil
}

// frameRunner reads image frames from framesChan and performs face and sentiment detections on them
// doneChan is used to receive a signal from the main goroutine to notify frameRunner to stop and return
func frameRunner(framesChan <-chan *gocv.Mat, doneChan <-chan struct{},
	resultsChan chan Result, mqttChan chan Result, faceNet, sentNet gocv.Net) error {
	// TODO: might make these globals/constants
	mean := gocv.NewScalar(0, 0, 0, 0)
	swapRB := false
	crop := false

	for {
		select {
		case <-doneChan:
			fmt.Printf("Stopping frameRunner: received stop sginal\n")
			return nil
		case frame := <-framesChan:
			// let's make a copy of the original
			img := gocv.NewMat()
			frame.CopyTo(&img)

			// convert img Mat to 672x384 blob that the face detector can analyze
			blob := gocv.BlobFromImage(img, 1.0, image.Pt(672, 384), mean, swapRB, crop)

			// feed the blob into the detector
			faceNet.SetInput(blob, "")

			// run a forward pass through the network
			results := faceNet.Forward("")

			// iterate through all detections and append results to faces buffer
			var faces []image.Rectangle
			for i := 0; i < results.Total(); i += 7 {
				confidence := results.GetFloatAt(0, i+2)
				if float64(confidence) > faceConfidence {
					left := int(results.GetFloatAt(0, i+3) * float32(img.Cols()))
					top := int(results.GetFloatAt(0, i+4) * float32(img.Rows()))
					right := int(results.GetFloatAt(0, i+5) * float32(img.Cols()))
					bottom := int(results.GetFloatAt(0, i+6) * float32(img.Rows()))
					faces = append(faces, image.Rect(left, top, right, bottom))
				}
			}

			sentMap := make(map[Sentiment]int)
			// do the sentiment detection here
			for i := range faces {
				// make sure the face rect is completely inside the main frame
				if !faces[i].In(image.Rect(0, 0, img.Cols(), img.Rows())) {
					continue
				}

				// propagate the detected face forward through sentiment network
				face := img.Region(faces[i])
				sentBlob := gocv.BlobFromImage(face, 1.0, image.Pt(64, 64), mean, swapRB, crop)
				sentNet.SetInput(sentBlob, "")
				sentResult := sentNet.Forward("")
				// flatten the result from [1, 5, 1, 1] to [1, 5]
				sentResult = sentResult.Reshape(1, 5)
				// find the most likely mood in returned list of sentiments
				_, confidence, _, maxLoc := gocv.MinMaxLoc(sentResult)

				var s Sentiment
				if float64(confidence) > sentConfidence {
					s = Sentiment(maxLoc.Y)
				} else {
					s = UNKNOWN
				}
				sentMap[s] += 1

				sentBlob.Close()
				sentResult.Close()
			}

			// send result down the channel
			result := Result{
				Shoppers:   len(faces),
				Detections: sentMap,
			}
			// send results down the results and mqtt channels
			resultsChan <- result
			mqttChan <- result
			img.Close()
			results.Close()
		}
	}

	return nil
}

func main() {
	// parse cli flags
	if err := parseCliFlags(); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing command line parameters: %v\n", err)
		os.Exit(1)
	}

	// read in Face model and set the backend and target
	faceNet := gocv.ReadNet(faceModel, faceConfig)
	if err := faceNet.SetPreferableBackend(gocv.NetBackendType(backend)); err != nil {
		fmt.Fprintf(os.Stderr, "Error setting backend for Face DNN: %v\n", err)
		os.Exit(1)
	}
	if err := faceNet.SetPreferableTarget(gocv.NetTargetType(target)); err != nil {
		fmt.Fprintf(os.Stderr, "Error setting target for Face DNN: %v\n", err)
		os.Exit(1)
	}

	// read in Snetiment model and set the backend and target
	sentNet := gocv.ReadNet(sentModel, sentConfig)
	if err := sentNet.SetPreferableBackend(gocv.NetBackendType(backend)); err != nil {
		fmt.Fprintf(os.Stderr, "Error setting backend for Sentiment DNN: %v\n", err)
		os.Exit(1)
	}
	if err := sentNet.SetPreferableTarget(gocv.NetTargetType(target)); err != nil {
		fmt.Fprintf(os.Stderr, "Error setting target for Sentiment DNN: %v\n", err)
		os.Exit(1)
	}

	// vc is videocapture source: either camera or file
	var vc *gocv.VideoCapture
	var err error
	if input != "" {
		// open camera device
		vc, err = gocv.VideoCaptureFile(input)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error opening video capture file: %v\n", input)
			os.Exit(1)
		}
	} else {
		// open camera device
		vc, err = gocv.VideoCaptureDevice(deviceID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error opening video capture device: %v\n", deviceID)
			os.Exit(1)
		}
	}
	defer vc.Close()

	// open display window
	window := gocv.NewWindow(name)
	defer window.Close()

	// prepare input image matrix
	img := gocv.NewMat()
	defer img.Close()

	fmt.Printf("Start reading frames from video capture source\n")

	// frames channel provides the source of images to process
	framesChan := make(chan *gocv.Mat, 1)
	// errChan is a channel used to capture program errors
	errChan := make(chan error, 2)
	// doneChan is used to signal goroutines they need to stop
	doneChan := make(chan struct{})
	// resultsChan is used for detection distribution
	resultsChan := make(chan Result, 1)
	// mqttChan is used for collecting MQTT stats
	mqttChan := make(chan Result, 1)
	// sigChan is used as a handler to stop all the goroutines
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, os.Kill, syscall.SIGTERM)

	// waitgroup to synchronise all goroutines
	var wg sync.WaitGroup

	// start frameRunner goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		errChan <- frameRunner(framesChan, doneChan, resultsChan, mqttChan, faceNet, sentNet)
	}()

	// create MQTT client and connect to MQTT server
	opts, err := MQTTClientOptions()
	if err != nil {
		fmt.Printf("%v\n", err)
		fmt.Printf("Have you set the ENV variables?\n")
	} else {
		mqttClient, err := MQTTConnect(opts)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error connecting to MQTT broker %s: %v\n",
				os.Getenv("MQTT_SERVER"), err)
			os.Exit(1)
		}
		// start MQTT worker goroutine
		wg.Add(1)
		go func() {
			defer wg.Done()
			errChan <- messageRunner(doneChan, mqttChan, mqttClient, mqttTopic, mqttRate)
		}()
		defer mqttClient.Disconnect(100)
	}

	var label string
detector:
	for {
		// read image frames from image capture source
		if ok := vc.Read(&img); !ok {
			fmt.Printf("Cannot read image source %v\n", deviceID)
			break
		}
		if img.Empty() {
			continue
		}

		// send image down the pipe for detection
		framesChan <- &img

		select {
		case sig := <-sigChan:
			fmt.Printf("Shutting down. Got signal: %s\n", sig)
			break detector
		case err = <-errChan:
			fmt.Printf("Shutting down. Failed with error: %s\n", err)
			break detector
		case result := <-resultsChan:
			label = fmt.Sprintf("%s", result)
			gocv.PutText(&img, label, image.Point{0, 40}, gocv.FontHersheySimplex, 0.5,
				color.RGBA{0, 0, 0, 0}, 2)
		}

		// show the image in the window, and wait 1 millisecond
		window.IMShow(img)
		window.WaitKey(1)
	}

	// signal to all goroutines to finish
	close(doneChan)
	// wait for all goroutines to finish
	wg.Wait()
}
