package com.example.plantflclient

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.plantflclient.ui.theme.PlantFLClientTheme
import java.io.File
import java.util.concurrent.Executors

// 🔥 UPDATE THIS IP: Run 'ipconfig' on your PC and use your IPv4 address here
const val SERVER_URL = "http://192.168.1.46:8000"
class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // 1. Try to download the latest model from the server in the background on startup
        val targetFile = File(filesDir, "updated_model.tflite")
        NetworkClient.downloadLatestModel(SERVER_URL, targetFile) { success ->
            if (success) {
                Log.d("FL_FLOW", "Latest model downloaded and saved to ${targetFile.absolutePath}")
            } else {
                Log.d("FL_FLOW", "Failed to download model or server offline. Using fallback.")
            }
        }

        // Test connection
        NetworkClient.testConnection(SERVER_URL)

        setContent {
            PlantFLClientTheme {

                var hasCameraPermission by remember {
                    mutableStateOf(
                        ContextCompat.checkSelfPermission(
                            this,
                            Manifest.permission.CAMERA
                        ) == PackageManager.PERMISSION_GRANTED
                    )
                }

                val launcher = rememberLauncherForActivityResult(
                    ActivityResultContracts.RequestPermission()
                ) { granted ->
                    hasCameraPermission = granted
                }

                LaunchedEffect(Unit) {
                    if (!hasCameraPermission) {
                        launcher.launch(Manifest.permission.CAMERA)
                    }
                }

                if (hasCameraPermission) {
                    CameraScreen()
                } else {
                    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        Text("Camera permission required")
                    }
                }
            }
        }
    }
}

@Composable
fun CameraScreen() {

    val context = LocalContext.current
    // tflite helper will handle choosing between assets and filesDir
    val tflite = remember { TFLiteHelper(context) }
    val datasetHelper = remember { LocalDatasetHelper(context) }

    var capturedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var capturedRotation by remember { mutableIntStateOf(0) }
    var result by remember { mutableStateOf("No prediction") }

    Column(modifier = Modifier.fillMaxSize()) {

        // Camera Preview
        Box(modifier = Modifier.weight(1f)) {
            CameraPreview { bitmap, rotation ->
                capturedBitmap = bitmap
                capturedRotation = rotation
            }
        }

        // Show captured image
        capturedBitmap?.let {
            Image(
                bitmap = it.asImageBitmap(),
                contentDescription = null,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(180.dp)
            )
        }

        Spacer(modifier = Modifier.height(10.dp))

        // Buttons
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {

            Button(onClick = {
                capturedBitmap?.let {
                    result = tflite.predict(it, capturedRotation)
                }
            }) {
                Text("Predict")
            }

            Button(onClick = {
                capturedBitmap?.let {
                    val prediction = tflite.predict(it, capturedRotation)
                    val label = prediction.substringBefore(" (")
                    val confidence = prediction.substringAfter("(").substringBefore(")").toFloatOrNull() ?: 0f

                    // 1. Send prediction info to server
                    NetworkClient.sendPrediction(
                        serverUrl = SERVER_URL,
                        label = label,
                        confidence = confidence
                    )
                    
                    // 2. Save image locally for real FL training in internal filesDir/local_dataset
                    datasetHelper.saveSample(it, label)
                    
                    Toast.makeText(context, "Image Saved & Info Uploaded!", Toast.LENGTH_SHORT).show()
                }
            }) {
                Text("Upload")
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        // Federated Learning Button
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Center
        ) {
            Button(
                onClick = {
                    Log.d("FL_FLOW", "BUTTON CLICKED - Initiating FL Round")
                    
                    val stats = datasetHelper.getDatasetStats()
                    val totalSamples = stats.values.sum()
                    
                    if (totalSamples == 0) {
                        Toast.makeText(context, "No local data! Use 'Upload' first.", Toast.LENGTH_SHORT).show()
                        return@Button
                    }

                    Toast.makeText(context, "Starting FL Round with $totalSamples samples...", Toast.LENGTH_SHORT).show()

                    // Step 1: Download global weights from server
                    NetworkClient.getGlobalWeightsBinary(SERVER_URL) { globalWeights ->
                        if (globalWeights == null) {
                            Log.e("FL_FLOW", "Weight download failed. Aborting.")
                            return@getGlobalWeightsBinary
                        }

                        try {
                            val flClient = FLClient(context)
                            
                            // Step 2: Perform Local Training
                            Log.d("FL_FLOW", "Training started on device...")
                            val updatedWeights = flClient.trainOnDevice(globalWeights)
                            Log.d("FL_FLOW", "Training done. Uploading weights...")
                            
                            // Step 3: Send Updated Binary Weights back to server
                            NetworkClient.sendWeightsBinary(
                                serverUrl = SERVER_URL,
                                clientId = "android_client",
                                roundId = 1,
                                numSamples = totalSamples,
                                weights = updatedWeights
                            ) {
                                (context as? MainActivity)?.runOnUiThread {
                                    Toast.makeText(context, "Round Complete! Weights uploaded.", Toast.LENGTH_LONG).show()
                                }
                            }
                        } catch (e: Exception) {
                            Log.e("FL_FLOW", "Error during FL: ${e.message}", e)
                        }
                    }
                },
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.secondary)
            ) {
                Text("Train & Send")
            }
        }

        Spacer(modifier = Modifier.height(10.dp))

        Text(
            text = result,
            modifier = Modifier.padding(16.dp)
        )
    }
}

@Composable
fun CameraPreview(onImageCaptured: (Bitmap, Int) -> Unit) {
    val context = LocalContext.current
    val previewView = remember { androidx.camera.view.PreviewView(context) }
    val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current
    val executor = remember { Executors.newSingleThreadExecutor() }
    var lastAnalysisTime by remember { mutableLongStateOf(0L) }

    AndroidView(
        factory = { previewView },
        modifier = Modifier.fillMaxSize()
    )

    LaunchedEffect(Unit) {
        val cameraProvider = androidx.camera.lifecycle.ProcessCameraProvider.getInstance(context).get()

        val preview = androidx.camera.core.Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }

        val imageAnalysis = androidx.camera.core.ImageAnalysis.Builder()
            .setBackpressureStrategy(androidx.camera.core.ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()

        imageAnalysis.setAnalyzer(executor) { imageProxy ->
            val currentTime = System.currentTimeMillis()
            if (currentTime - lastAnalysisTime >= 1000) {
                val rotation = imageProxy.imageInfo.rotationDegrees
                val bitmap = imageProxy.toBitmap()
                onImageCaptured(bitmap, rotation)
                lastAnalysisTime = currentTime
            }
            imageProxy.close()
        }

        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(lifecycleOwner, androidx.camera.core.CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}
