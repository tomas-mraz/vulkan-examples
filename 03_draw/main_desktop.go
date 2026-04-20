//go:build !android

package main

import (
	"log"
	"runtime"

	"github.com/tomas-mraz/vulkan-ash"
)

const (
	windowWidth  = 640
	windowHeight = 480
)

func init() {
	runtime.LockOSThread()
}

func start() {
	host := ash.NewDesktopHost(windowWidth, windowHeight, appName)
	session := ash.NewSession(host, appName, nil)

	if err := session.Run(&drawRenderer{}); err != nil {
		log.Fatal(err)
	}
}
