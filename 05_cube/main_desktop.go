//go:build !android

package main

import (
	"log"
	"runtime"

	ash "github.com/tomas-mraz/vulkan-ash"
)

const (
	windowWidth  = 500
	windowHeight = 500
)

func init() {
	runtime.LockOSThread()
}

func start() {
	host := ash.NewDesktopHost(windowWidth, windowHeight, appName)
	session := ash.NewSession(host, appName, nil)

	if err := session.Run(&cubeRenderer{}); err != nil {
		log.Fatal(err)
	}
}
