package com.margelo.nitro.nitroonnxruntime

import android.annotation.SuppressLint
import android.content.Context
import android.net.Uri
import android.util.Log
import com.facebook.proguard.annotations.DoNotStrip
import com.margelo.nitro.core.ArrayBuffer
import com.margelo.nitro.core.Promise
import com.margelo.nitro.NitroModules
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.nio.ByteBuffer
import java.util.Objects
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response

//Code from thx mrousavy (https://github.com/mrousavy/react-native-fast-tflite/blob/main/android/src/main/java/com/tflite/TfliteModule.java)
@DoNotStrip
class AssetManager : HybridAssetManagerSpec() {
  companion object {
    private const val TAG = "AssetManager"
    private val client = OkHttpClient()

    // @SuppressLint("DiscouragedApi")
    // private fun getResourceId(context: Context, name: String): Int {
    //   return context.resources.getIdentifier(
    //     name,
    //     "raw",
    //     context.packageName
    //   )
    // }
  }

  override fun copyFile(source: String): Promise<String> {
      return Promise.async {
          try {
              Log.i(TAG, "Copying file from source: $source...")

              val context = NitroModules.applicationContext
                  ?: throw Error("Application context is unavailable")

              // Extract the base filename without query parameters
              val baseFileName = try {
                  if (source.contains("?")) {
                      File(source.substring(0, source.indexOf("?"))).name
                  } else {
                      File(source).name
                  }
              } catch (e: IllegalArgumentException) {
                  throw Error("Invalid source path format: $source - ${e.message}")
              }

              // Create destination file
              val destinationFile = File(context.getExternalFilesDir(null), baseFileName)

              if (!destinationFile.parentFile?.exists()!!) {
                  destinationFile.parentFile?.mkdirs()
                      ?: throw Error("Failed to create destination directory")
              }

              if (!destinationFile.exists()) {
                  Log.i(TAG, "File doesn't exist, copying from source...")

                  when {
                      source.contains("://") -> {
                          val uri = Uri.parse(source)
                          when (uri.scheme) {
                              "file" -> copyLocalFile(uri, destinationFile)
                              "http", "https" -> downloadNetworkFile(source, destinationFile)
                              else -> throw Error("Unsupported URI scheme: ${uri.scheme}")
                          }
                      }
                      // else -> copyFromAssets(context, source, destinationFile)
                      else -> throw Error("Unsupported source type: $source")
                  }

                  Log.i(TAG, "File copied successfully to: ${destinationFile.absolutePath}")
              } else {
                  Log.i(TAG, "File already exists at destination: ${destinationFile.absolutePath}")
              }

              destinationFile.absolutePath

          } catch (e: Error) {
              Log.e(TAG, "Error copying file: ${e.message}", e)
              throw e  // Re-throw the original Error
          } catch (e: Exception) {
              // Wrap any unexpected exceptions in Error for hybrid compatibility
              val errorMessage = "Unexpected error copying file from $source: ${e.message}"
              Log.e(TAG, errorMessage, e)
              throw Error(errorMessage)
          }
      }
  }

  private fun copyLocalFile(uri: Uri, destinationFile: File) {
      try {
          FileInputStream(File(uri.path ?: throw Error("Invalid file URI")))
              .use { input ->
                  FileOutputStream(destinationFile).use { output ->
                      input.copyTo(output)
                  }
              }
      } catch (e: Exception) {
          throw Error("Failed to copy local file: ${e.message}")
      }
  }

  private fun downloadNetworkFile(source: String, destinationFile: File) {
      try {
          val request = Request.Builder().url(source).build()
          client.newCall(request).execute().use { response ->
              if (!response.isSuccessful) {
                  throw Error("HTTP error ${response.code} downloading file from $source")
              }
              response.body?.byteStream()?.let { input ->
                  FileOutputStream(destinationFile).use { output ->
                      input.copyTo(output)
                  }
              } ?: throw Error("Empty response body from $source")
          }
      } catch (e: Exception) {
          throw Error("Failed to download network file: ${e.message}")
      }
  }

  // private fun copyFromAssets(context: Context, source: String, destinationFile: File) {
  //     try {
  //         context.assets.open(source).use { input ->
  //             FileOutputStream(destinationFile).use { output ->
  //                 input.copyTo(output)
  //             }
  //         }
  //     } catch (e: Exception) {
  //         throw Error("Failed to copy from assets: ${e.message}")
  //     }
  // }
}
