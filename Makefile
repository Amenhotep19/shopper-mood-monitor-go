BUILD=go build
CLEAN=go clean
INSTALL=go install
BUILDPATH=./build
PACKAGES=$(shell go list ./... )

CGO_CXXFLAGS="--std=c++11"
CGO_CPPFLAGS="-I${INTEL_CVSDK_DIR}/opencv/include -I${INTEL_CVSDK_DIR}/deployment_tools/inference_engine/include"
CGO_LDFLAGS="-L${INTEL_CVSDK_DIR}/opencv/lib -L${INTEL_CVSDK_DIR}/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64 -lpthread -ldl -ldla -ldliaPlugin -lHeteroPlugin -lMKLDNNPlugin -lmyriadPlugin -linference_engine -lclDNNPlugin -lopencv_core -lopencv_pvl -lopencv_face -lopencv_videoio -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_objdetect -lopencv_features2d -lopencv_video -lopencv_dnn -lopencv_calib3d"
PKG_CONFIG_PATH=/usr/lib64/pkgconfig

all: test build

build: dir
	go build -o "$(BUILDPATH)/monitor" "main.go"

dir:
	mkdir -p $(BUILDPATH)

install:
	$(INSTALL) ./$(EXDIR)/...

clean:
	rm -rf $(BUILDPATH)/*

godep:
	go get -u github.com/golang/dep/cmd/dep

dep: godep
	dep ensure

check:
	for pkg in ${PACKAGES}; do \
		go vet $$pkg || exit ; \
		golint $$pkg || exit ; \
	done

test:
	for pkg in ${PACKAGES}; do \
		go test -coverprofile="../../../$$pkg/coverage.txt" -covermode=atomic $$pkg || exit; \
	done

.PHONY: clean build all
