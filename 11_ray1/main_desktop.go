//go:build !android

package main

import (
	"log"
	"runtime"

	ash "github.com/tomas-mraz/vulkan-ash"
)

const (
	windowWidth  = 800
	windowHeight = 600
)

func init() {
	runtime.LockOSThread()
}

func start() {
	if err := run(ash.NewDesktopHost(windowWidth, windowHeight, appName)); err != nil {
		log.Fatal(err)
	}
}
