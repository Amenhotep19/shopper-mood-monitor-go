/*
* Copyright (c) 2018 Intel Corporation.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
* LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
* OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
* WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

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
	topic = "retail/traffic"
)

var (
	// deviceID is camera device ID
	deviceID int
	// input is path to image or video file
	input string
	// faceModel is path to .bin file of face detection model
	faceModel string
	// faceConfig is path to .xml file of face detection model configuration
	faceConfig string
	// faceConfidence is confidence threshold for face detection model
	faceConfidence float64
	// sentModel is path to .bin file of sentiment detection model
	sentModel string
	// sentConfig is path to .xml of sentiment detection model configuration
	sentConfig string
	// sentConfidence is confidence threshold for sentiment detection model
	sentConfidence float64
	// backend is inference backend
	backend int
	// target is inference target
	target int
	// publish is a flag which instructs the program to publish data analytics
	publish bool
	// rate is number of seconds between analytics are collected and sent to a remote server
	rate int
	// delay is video playback delay
	delay float64
)

func init() {
	flag.IntVar(&deviceID, "device", 0, "Camera device ID")
	flag.StringVar(&input, "input", "", "Path to image or video file")
	flag.StringVar(&faceModel, "face-model", "", "Path to .bin file of face detection model")
	flag.StringVar(&faceConfig, "face-config", "", "Path to .xml file of face model configuration")
	flag.Float64Var(&faceConfidence, "face-confidence", 0.5, "Confidence threshold for face detection")
	flag.StringVar(&sentModel, "sent-model", "", "Path to .bin file of sentiment detection model")
	flag.StringVar(&sentConfig, "sent-config", "", "Path to .xml file of sentiment model configuration")
	flag.Float64Var(&sentConfidence, "sent-confidence", 0.5, "Confidence threshold for sentiment detection")
	flag.IntVar(&backend, "backend", 0, "Inference backend. 0: Auto, 1: Halide language, 2: Intel DL Inference Engine")
	flag.IntVar(&target, "target", 0, "Target device. 0: CPU, 1: OpenCL, 2: OpenCL half precision, 3: VPU")
	flag.BoolVar(&publish, "publish", false, "Publish data analytics to a remote server")
	flag.IntVar(&rate, "rate", 1, "Number of seconds between analytics are sent to a remote server")
	flag.Float64Var(&delay, "delay", 5.0, "Video playback delay")
}

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
	// ANGRY  is for detecting anger emotion
	ANGRY
	// UNKNOWN is catchall unidentifiable emotion
	UNKNOWN
)

// String implements fmt.Stringer interface for Sentiment.
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
	case ANGRY:
		return "ANGRY"
	default:
		return "UNKNOWN"
	}
}

// Perf stores inference engine performance info.
type Perf struct {
	// FaceNet stores face detector performance info.
	FaceNet float64
	// SentNet stores sentiment detector performance info.
	SentNet float64
}

// String implements fmt.Stringer interface for Perf.
func (p *Perf) String() string {
	return fmt.Sprintf("Face inference time: %.2f ms, Sentiment inference time: %.2f ms", p.FaceNet, p.SentNet)
}

// Result is inference result.
type Result struct {
	// Shoppers contains count of all detected faces of shoppers.
	Shoppers int
	// Detections is a map which contains detected sentiment of each shopper.
	Detections map[Sentiment]int
	// Perf is DNN inference speed performance.
	Perf *Perf
}

// initDetectionsMap initializes Result Detections map to zero values.
func (r *Result) initDetectionsMap() {
	r.Detections = make(map[Sentiment]int)
	r.Detections[NEUTRAL] = 0
	r.Detections[HAPPY] = 0
	r.Detections[SAD] = 0
	r.Detections[SURPRISED] = 0
	r.Detections[ANGRY] = 0
	r.Detections[UNKNOWN] = 0
}

// String implements fmt.Stringer interface for Result.
func (r *Result) String() string {
	// if the map is empty, initialize all keys to zero values
	if r.Detections == nil {
		r.initDetectionsMap()
	}

	return fmt.Sprintf("Shoppers: %d, Neutral: %d, Happy: %d, Sad: %d, Surprised: %d, Anger: %d, Unknown: %d",
		r.Shoppers, r.Detections[NEUTRAL], r.Detections[HAPPY], r.Detections[SAD],
		r.Detections[SURPRISED], r.Detections[ANGRY], r.Detections[UNKNOWN])
}

// ToMQTTMessage turns result into MQTT message which can be published to MQTT broker.
func (r *Result) ToMQTTMessage() string {
	// if the map is empty, initialize all keys to zero values
	if r.Detections == nil {
		r.initDetectionsMap()
	}

	return fmt.Sprintf("{\"shoppers\": \"%d\", \"neutral\": \"%d\", \"happy\": \"%d\", \"sad\": \"%d\", \"surprised\": \"%d\", \"anger\": \"%d\", \"unknown\": \"%d\"}",
		r.Shoppers, r.Detections[NEUTRAL], r.Detections[HAPPY], r.Detections[SAD],
		r.Detections[SURPRISED], r.Detections[ANGRY], r.Detections[UNKNOWN])
}

// getPerformanceInfo queries the Inference Engine performance info and returns it as string.
func getPerformanceInfo(faceNet, sentNet *gocv.Net, sentChecked bool) *Perf {
	freq := gocv.GetTickFrequency() / 1000

	facePerf := faceNet.GetPerfProfile() / freq
	var sentPerf float64
	if sentChecked {
		sentPerf = sentNet.GetPerfProfile() / freq
	}

	return &Perf{
		FaceNet: facePerf,
		SentNet: sentPerf,
	}
}

// messageRunner reads data published to pubChan with rate frequency and sends them to remote analytics server
// doneChan is used to receive a signal from the main goroutine to notify the routine to stop and return
func messageRunner(doneChan <-chan struct{}, pubChan <-chan *Result, c *MQTTClient, topic string, rate int) error {
	ticker := time.NewTicker(time.Duration(rate) * time.Second)

	for {
		select {
		case <-ticker.C:
			result := <-pubChan
			_, err := c.Publish(topic, result.ToMQTTMessage())
			// Just signal there was an error and carry on
			if err != nil {
				fmt.Printf("Error publishing message to %s: %v", topic, err)
			}
		case <-pubChan:
			// we discard messages in between ticker times
		case <-doneChan:
			fmt.Printf("Stopping messageRunner: received stop signal\n")
			return nil
		}
	}
}

// detectSentiment detects sentiment in img regions defined by faces rectangles
// It returns a map of counts of detected sentiment for each faces.
func detectSentiment(net *gocv.Net, img *gocv.Mat, faces []image.Rectangle) map[Sentiment]int {
	// sentMap maps the counts of each detected face emotion
	sentMap := make(map[Sentiment]int)
	// do the sentiment detection here
	for i := range faces {
		// make sure the face rect is completely inside the main frame
		if !faces[i].In(image.Rect(0, 0, img.Cols(), img.Rows())) {
			continue
		}

		// propagate the detected face forward through sentiment network
		face := img.Region(faces[i])
		blob := gocv.BlobFromImage(face, 1.0, image.Pt(64, 64), gocv.NewScalar(0, 0, 0, 0), false, false)

		// run a forward pass through the network
		net.SetInput(blob, "")
		result := net.Forward("")

		// flatten the result from [1, 5, 1, 1] to [1, 5]
		result = result.Reshape(1, 5)

		// find the most likely mood in returned list of sentiments
		_, confidence, _, maxLoc := gocv.MinMaxLoc(result)

		var s Sentiment
		if float64(confidence) > sentConfidence {
			s = Sentiment(maxLoc.Y)
		} else {
			s = UNKNOWN
		}
		sentMap[s]++

		blob.Close()
		result.Close()
	}

	return sentMap
}

// detectFaces detects faces in img and returns them as a slice of rectangles that encapsulates them
func detectFaces(net *gocv.Net, img *gocv.Mat) []image.Rectangle {
	// convert img Mat to 672x384 blob that the face detector can analyze
	blob := gocv.BlobFromImage(*img, 1.0, image.Pt(672, 384), gocv.NewScalar(0, 0, 0, 0), false, false)
	defer blob.Close()

	// run a forward pass through the network
	net.SetInput(blob, "")
	results := net.Forward("")
	defer results.Close()

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

	return faces
}

// frameRunner reads image frames from framesChan and performs face and sentiment detections on them
// doneChan is used to receive a signal from the main goroutine to notify frameRunner to stop and return
func frameRunner(framesChan <-chan *frame, doneChan <-chan struct{}, resultsChan chan<- *Result,
	pubChan chan<- *Result, faceNet, sentNet *gocv.Net) error {

	// frame is image frame
	// we want to avoid continuous allocation that lead to GC pauses
	frame := new(frame)

	// allocate perf once
	perf := new(Perf)

	for {
		select {
		case <-doneChan:
			fmt.Printf("Stopping frameRunner: received stop signal\n")
			close(resultsChan)
			if pubChan != nil {
				close(pubChan)
			}
			return nil
		case frame = <-framesChan:
			if frame == nil {
				continue
			}

			// let's make a copy of the original
			img := gocv.NewMat()
			frame.img.CopyTo(&img)

			// detect faces and return them
			faces := detectFaces(faceNet, &img)

			// sentChecked is only set to true if at least one sentiment has been detected
			sentChecked := false

			// detect sentiment from detected faces
			sentMap := detectSentiment(sentNet, &img, faces)

			if len(sentMap) != 0 {
				sentChecked = true
			}

			perf = getPerformanceInfo(faceNet, sentNet, sentChecked)

			// detection result
			result := &Result{
				Shoppers:   len(faces),
				Detections: sentMap,
				Perf:       perf,
			}

			// send data down the channels
			resultsChan <- result
			if pubChan != nil {
				pubChan <- result
			}

			// close image matrices
			img.Close()
		}
	}
}

func parseCliFlags() error {
	flag.Parse()

	// path to face detection model can't be empty
	if faceModel == "" {
		return fmt.Errorf("Invalid path to .bin file of face detection model: %s", faceModel)
	}
	// path to face detection model config can't be empty
	if faceConfig == "" {
		return fmt.Errorf("Invalid path to .xml file of face model configuration: %s", faceConfig)
	}
	// path to sentiment detection model can't be empty
	if sentModel == "" {
		return fmt.Errorf("Invalid path to .bin file of sentiment detection model: %s", sentModel)
	}
	// path to sentiment detection model config can't be empty
	if sentConfig == "" {
		return fmt.Errorf("Invalid path to .xml file of sentiment model configuration: %s", sentConfig)
	}

	return nil
}

// NewInferModel reads DNN model and it configuration, sets its preferable target and backend and returns it.
// It returns error if either the model files failed to be read or setting the target fails
func NewInferModel(model, config string, backend, target int) (*gocv.Net, error) {
	m := gocv.ReadNet(model, config)

	if err := m.SetPreferableBackend(gocv.NetBackendType(backend)); err != nil {
		return nil, err
	}

	if err := m.SetPreferableTarget(gocv.NetTargetType(target)); err != nil {
		return nil, err
	}

	return &m, nil
}

// NewCapture creates new video capture from input or camera backend if input is empty and returns it.
// If input is not empty, NewCapture adjusts delay parameter so video playback matches FPS in the video file.
// It fails with error if it either can't open the input video file or the video device
func NewCapture(input string, deviceID int, delay *float64) (*gocv.VideoCapture, error) {
	if input != "" {
		// open video file
		vc, err := gocv.VideoCaptureFile(input)
		if err != nil {
			return nil, err
		}

		fps := vc.Get(gocv.VideoCaptureFPS)
		*delay = 1000 / fps

		return vc, nil
	}

	// open camera device
	vc, err := gocv.VideoCaptureDevice(deviceID)
	if err != nil {
		return nil, err
	}

	return vc, nil
}

// NewMQTTPublisher creates new MQTT client by reading environment variables anrd returns it.
// It attempts to make a connection to the remote server and if successful it returns the client handler.
// It returns error if either the connection to the remote server failed or if the client config is invalid.
func NewMQTTPublisher() (*MQTTClient, error) {
	opts, err := MQTTClientOptions()
	if err != nil {
		return nil, err
	}

	c, err := MQTTConnect(opts)
	if err != nil {
		return nil, err
	}

	return c, nil
}

// frame is used to send video frames and program configuration to upstream goroutines
type frame struct {
	// img is image frame
	img *gocv.Mat
}

func main() {
	// parse cli flags
	if err := parseCliFlags(); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing command line parameters: %v\n", err)
		os.Exit(1)
	}

	// read in Face model and set its inference backend and target
	faceNet, err := NewInferModel(faceModel, faceConfig, backend, target)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating Face detection model: %v\n", err)
		os.Exit(1)
	}

	// read in Sentiment model and set its inference backend and target
	sentNet, err := NewInferModel(sentModel, sentConfig, backend, target)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating Sentiment detection model: %v\n", err)
		os.Exit(1)
	}

	// create new video capture
	vc, err := NewCapture(input, deviceID, &delay)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating new video capture: %v\n", err)
		os.Exit(1)
	}
	defer vc.Close()

	// frames channel provides the source of images to process
	framesChan := make(chan *frame, 1)

	// errChan is a channel used to capture program errors
	errChan := make(chan error, 2)

	// doneChan is used to signal goroutines they need to stop
	doneChan := make(chan struct{})

	// resultsChan is used for detection distribution
	resultsChan := make(chan *Result, 1)

	// sigChan is used as a handler to stop all the goroutines
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, os.Kill, syscall.SIGTERM)

	// pubChan is used for publishing data analytics stats
	var pubChan chan *Result

	// waitgroup to synchronize all goroutines
	var wg sync.WaitGroup

	if publish {
		p, err := NewMQTTPublisher()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to create MQTT publisher: %v\n", err)
			os.Exit(1)
		}
		pubChan = make(chan *Result, 1)

		// start MQTT worker goroutine
		wg.Add(1)
		go func() {
			defer wg.Done()
			errChan <- messageRunner(doneChan, pubChan, p, topic, rate)
		}()
		defer p.Disconnect(100)
	}

	// start frameRunner goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		errChan <- frameRunner(framesChan, doneChan, resultsChan, pubChan, faceNet, sentNet)
	}()

	// open display window
	window := gocv.NewWindow(name)
	window.SetWindowProperty(gocv.WindowPropertyAutosize, gocv.WindowAutosize)
	defer window.Close()

	// prepare input image matrix
	img := gocv.NewMat()
	defer img.Close()

	// initialize the result pointers
	result := new(Result)

monitor:
	for {
		if ok := vc.Read(&img); !ok {
			fmt.Printf("Cannot read image source %v\n", deviceID)
			break
		}
		if img.Empty() {
			continue
		}

		framesChan <- &frame{img: &img}

		select {
		case sig := <-sigChan:
			fmt.Printf("Shutting down. Got signal: %s\n", sig)
			break monitor
		case err = <-errChan:
			fmt.Printf("Shutting down. Encountered error: %s\n", err)
			break monitor
		case result = <-resultsChan:
		default:
			// do nothing; just display latest results
		}

		// inference performance and print it
		gocv.PutText(&img, fmt.Sprintf("%s", result.Perf), image.Point{0, 15},
			gocv.FontHersheySimplex, 0.5, color.RGBA{0, 0, 0, 0}, 2)

		// inference results label
		gocv.PutText(&img, fmt.Sprintf("%s", result), image.Point{0, 40},
			gocv.FontHersheySimplex, 0.5, color.RGBA{0, 0, 0, 0}, 2)

		// show the image in the window, and wait 1 millisecond
		window.IMShow(img)

		// exit when ESC key is pressed
		if window.WaitKey(int(delay)) == 27 {
			fmt.Printf("Attempting to shut down: ESC pressed\n")
			break monitor
		}
	}

	// signal all goroutines to finish
	close(framesChan)
	close(doneChan)

	// unblock frameRunner by emptying resultsChan if need be
	for range resultsChan {
		// collect any outstanding results
	}

	// wait for all goroutines to finish
	wg.Wait()
}
