package com.example.plantflclient

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.util.*

class LocalDatasetHelper(private val context: Context) {

    private val datasetDir = File(context.filesDir, "local_dataset")

    init {
        if (!datasetDir.exists()) {
            datasetDir.mkdirs()
        }
    }

    /**
     * Saves a captured image and its label for future on-device training.
     */
    fun saveSample(bitmap: Bitmap, label: String) {
        try {
            val timestamp = System.currentTimeMillis()
            val filename = "${label}_$timestamp.jpg"
            val file = File(datasetDir, filename)

            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
            }
            
            Log.d("DATASET", "Saved training sample: $filename")
        } catch (e: Exception) {
            Log.e("DATASET", "Error saving sample: ${e.message}")
        }
    }

    /**
     * Returns the count of saved samples for each class.
     */
    fun getDatasetStats(): Map<String, Int> {
        val files = datasetDir.listFiles() ?: return emptyMap()
        return files.groupBy { it.name.substringBefore("_") }
            .mapValues { it.value.size }
    }

    /**
     * Clears the local dataset after a training round is complete.
     */
    fun clearDataset() {
        datasetDir.listFiles()?.forEach { it.delete() }
        Log.d("DATASET", "Local dataset cleared.")
    }
}
