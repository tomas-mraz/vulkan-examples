//go:build android

package main

import (
	"log"

	ash "github.com/tomas-mraz/vulkan-ash"

	"github.com/tomas-mraz/android-go/app"
)

func start() {
	app.Main(func(a app.NativeActivity) {
		if err := run(ash.NewAndroidHost(a)); err != nil {
			log.Fatal(err)
		}
	})
}
