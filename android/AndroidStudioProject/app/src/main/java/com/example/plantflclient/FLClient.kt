package com.example.plantflclient

import android.content.Context
import kotlin.random.Random

class FLClient(private val context: Context) {

    /**
     * Performs "training" on a FloatArray. 
     * This avoids JSON overhead entirely.
     */
    fun trainOnDevice(globalWeights: FloatArray): FloatArray {
        // We update the weights in-place or create a new array.
        // For 2.5M floats, this uses exactly 10MB of RAM.
        val updated = FloatArray(globalWeights.size)

        for (i in globalWeights.indices) {
            val w = globalWeights[i]
            // Simulate training (small update)
            updated[i] = w + Random.nextDouble(-0.01, 0.01).toFloat()
        }

        return updated
    }
}
