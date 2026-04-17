//go:build android

package main

import (
	"log"

	ash "github.com/tomas-mraz/vulkan-ash"

	"github.com/tomas-mraz/android-go/app"
)

func start() {
	app.Main(func(a app.NativeActivity) {
		host := ash.NewAndroidHost(a)
		session := ash.NewSession(host, appName, nil)

		if err := session.Run(&cubeRenderer{}); err != nil {
			log.Fatal(err)
		}
	})
}
