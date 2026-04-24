package com.example.plantflclient

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

class TFLiteHelper(context: Context) {

    private var interpreter: Interpreter? = null
    private val labels: List<String>
    private val imageSize = 224

    init {
        val localFile = File(context.filesDir, "updated_model.tflite")
        
        val modelBuffer = if (localFile.exists() && localFile.length() > 0) {
            try {
                Log.d("TFLiteHelper", "Loading downloaded model: ${localFile.length()} bytes")
                loadModelFromFile(localFile)
            } catch (e: Exception) {
                Log.e("TFLiteHelper", "Downloaded model failed to load, falling back", e)
                localFile.delete()
                loadModelFromAssets(context, "model.tflite")
            }
        } else {
            Log.d("TFLiteHelper", "Loading bundled model from assets")
            loadModelFromAssets(context, "model.tflite")
        }

        try {
            interpreter = Interpreter(modelBuffer, Interpreter.Options())
            Log.d("TFLiteHelper", "Interpreter initialized successfully")
        } catch (e: Exception) {
            Log.e("TFLiteHelper", "Interpreter init failed. Final fallback to assets.", e)
            interpreter = Interpreter(loadModelFromAssets(context, "model.tflite"), Interpreter.Options())
        }

        labels = context.assets.open("labels.txt")
            .bufferedReader()
            .readLines()
            .map { it.trim() }
            .filter { it.isNotEmpty() }
    }

    private fun loadModelFromFile(file: File): ByteBuffer {
        val inputStream = FileInputStream(file)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
    }

    private fun loadModelFromAssets(context: Context, modelName: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    fun predict(bitmap: Bitmap, rotation: Int): String {
        val interp = interpreter ?: return "Model not ready"
        try {
            val rotations = ((360 - (rotation % 360)) / 90) % 4
            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(imageSize, imageSize, ResizeOp.ResizeMethod.BILINEAR))
                .build()

            val tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)
            val processedImage = imageProcessor.process(tensorImage)

            val output = Array(1) { FloatArray(labels.size) }
            interp.run(processedImage.buffer, output)

            val probs = output[0]
            
            // Log raw output for debugging NaN issues
            Log.d("TFLiteHelper", "Raw Probs: ${probs.take(3).joinToString()} ...")

            val maxIndex = probs.indices.maxByOrNull { probs[it] } ?: return "Unknown"
            val confidence = probs[maxIndex]

            // Handle NaN
            if (confidence.isNaN()) {
                Log.e("TFLiteHelper", "Model output NaN! Check weights on server.")
                return "${labels[maxIndex]} (0.00)"
            }

            return "${labels[maxIndex]} (${String.format("%.2f", confidence)})"
        } catch (e: Exception) {
            Log.e("TFLiteHelper", "Prediction failed", e)
            return "Unknown"
        }
    }
}
