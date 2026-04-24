package com.example.plantflclient

import android.util.Log
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

object NetworkClient {

    private val client = OkHttpClient()

    fun sendPrediction(serverUrl: String, label: String, confidence: Float) {
        val json = JSONObject().apply {
            put("label", label)
            put("confidence", confidence)
        }
        val body = json.toString().toRequestBody("application/json".toMediaTypeOrNull())
        val request = Request.Builder().url("$serverUrl/upload_prediction").post(body).build()
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) { Log.e("NetworkClient", "Fail: ${e.message}") }
            override fun onResponse(call: Call, response: Response) { response.close() }
        })
    }

    fun getGlobalWeightsBinary(serverUrl: String, callback: (FloatArray?) -> Unit) {
        Log.d("FL_FLOW", "Requesting global weights...")
        val request = Request.Builder()
            .url("$serverUrl/get_global_weights_binary")
            .get()
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e("FL_FLOW", "Download failed: ${e.message}")
                callback(null)
            }

            override fun onResponse(call: Call, response: Response) {
                try {
                    val bytes = response.body?.bytes()
                    if (!response.isSuccessful || bytes == null) {
                        Log.e("FL_FLOW", "Server error or empty body: ${response.code}")
                        callback(null)
                        return
                    }
                    val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                    val floatArray = FloatArray(bytes.size / 4)
                    for (i in floatArray.indices) {
                        floatArray[i] = buffer.float
                    }
                    callback(floatArray)
                } catch (e: Exception) {
                    Log.e("FL_FLOW", "Error parsing binary weights: ${e.message}")
                    callback(null)
                } finally {
                    response.close()
                }
            }
        })
    }

    fun sendWeightsBinary(
        serverUrl: String,
        clientId: String,
        roundId: Int,
        numSamples: Int,
        weights: FloatArray,
        onComplete: () -> Unit
    ) {
        val byteBuffer = ByteBuffer.allocate(weights.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (w in weights) { byteBuffer.putFloat(w) }

        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("client_id", clientId)
            .addFormDataPart("round_id", roundId.toString())
            .addFormDataPart("num_samples", numSamples.toString())
            .addFormDataPart(
                "weights_file", 
                "weights.bin",
                byteBuffer.array().toRequestBody("application/octet-stream".toMediaTypeOrNull())
            )
            .build()

        val request = Request.Builder().url("$serverUrl/upload_weights_binary").post(requestBody).build()
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) { onComplete() }
            override fun onResponse(call: Call, response: Response) { onComplete(); response.close() }
        })
    }

    /**
     * Downloads the latest TFLite model safely using a temporary file.
     */
    fun downloadLatestModel(serverUrl: String, targetFile: File, callback: (Boolean) -> Unit) {
        val request = Request.Builder().url("$serverUrl/get_model").get().build()
        val tempFile = File(targetFile.absolutePath + ".tmp")

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e("NetworkClient", "Download failed: ${e.message}")
                callback(false)
            }

            override fun onResponse(call: Call, response: Response) {
                if (!response.isSuccessful) {
                    callback(false)
                    response.close()
                    return
                }
                try {
                    val body = response.body ?: throw IOException("Empty body")
                    tempFile.outputStream().use { output ->
                        body.byteStream().use { input ->
                            input.copyTo(output)
                        }
                    }
                    // Atomic rename: TFLiteHelper won't see a half-finished file anymore
                    if (tempFile.renameTo(targetFile)) {
                        callback(true)
                    } else {
                        Log.e("NetworkClient", "Failed to rename temp model file")
                        callback(false)
                    }
                } catch (e: Exception) {
                    Log.e("NetworkClient", "Error saving model: ${e.message}")
                    callback(false)
                } finally {
                    response.close()
                }
            }
        })
    }

    fun triggerAggregation(serverUrl: String, onComplete: () -> Unit) {
        val body = "".toRequestBody(null)
        val request = Request.Builder().url("$serverUrl/aggregate").post(body).build()
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) { onComplete() }
            override fun onResponse(call: Call, response: Response) { onComplete(); response.close() }
        })
    }

    fun testConnection(serverUrl: String) {
        val request = Request.Builder().url("$serverUrl/health").get().build()
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) { Log.e("TEST", "Offline") }
            override fun onResponse(call: Call, response: Response) { response.close() }
        })
    }
}
