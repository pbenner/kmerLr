
VERSION   = 1.0.0
FILES     = kmerLr.go $(filter-out %_test.go kmerLr.go,$(wildcard *.go))
FILES_DEP = kmerLr.go $(filter-out %_test.go kmerLr.go,$(wildcard *.go config/*.go))
GOBIN     = $(shell echo $${GOPATH}/bin)

# ------------------------------------------------------------------------------

all: kmerLr

kmerLr: $(FILES_DEP)
	go build -ldflags "\
	   -X main.Version=$(VERSION) \
	   -X main.BuildTime=`TZ=UTC date -u '+%Y-%m-%dT%H:%M:%SZ'` \
	   -X main.GitHash=`git rev-parse HEAD`" \
	   $(FILES)

install: kmerLr | $(GOBIN)
ifeq ($(GOBIN),/bin)
	install kmerLr $$HOME/go/bin
else
	install kmerLr $(GOBIN)
endif

$(GOBIN):
	mkdir -p $(GOBIN)

# ------------------------------------------------------------------------------

.PHONY: all install
