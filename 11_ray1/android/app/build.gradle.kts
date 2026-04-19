plugins {
    id("com.android.application")
}

android {
    namespace = "com.vulkanexamples.ray1"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.vulkanexamples.ray1"
        minSdk = 34
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    splits {
        abi {
            isEnable = true
            reset()
            include("arm64-v8a", "x86_64")
            isUniversalApk = false
        }
    }

    packaging {
        jniLibs {
            keepDebugSymbols += "**/*.so"
        }
    }
}
