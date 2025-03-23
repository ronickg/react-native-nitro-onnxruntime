package com.margelo.nitro.nitroonnxruntime

import android.annotation.SuppressLint
import android.content.Context
import android.net.Uri
import android.util.Log
import com.facebook.proguard.annotations.DoNotStrip
import com.margelo.nitro.core.ArrayBuffer
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStream
import java.lang.ref.WeakReference
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
    private var weakContext: WeakReference<Context>? = null

    fun setContext(context: Context) {
      weakContext = WeakReference(context)
    }

    @SuppressLint("DiscouragedApi")
    private fun getResourceId(context: Context, name: String): Int {
      return context.resources.getIdentifier(
        name,
        "raw",
        context.packageName
      )
    }

    private fun getLocalFileBytes(stream: InputStream, file: File): ByteArray {
      val fileSize = file.length()

      if (fileSize > Integer.MAX_VALUE) {
        throw IOException("File is too large to read into memory")
      }

      val data = ByteArray(fileSize.toInt())

      var bytesRead = 0
      var chunk: Int = 0
      while (bytesRead < fileSize && stream.read(data, bytesRead, (fileSize - bytesRead).toInt()).also { chunk = it } != -1) {
        bytesRead += chunk
      }

      if (bytesRead != fileSize.toInt()) {
        throw IOException("Could not completely read file ${file.name}")
      }

      return data
    }
  }

  override fun fetchByteDataFromUrl(url: String): ArrayBuffer {
    try {
      Log.i(TAG, "Loading byte data from URL: $url...")

      var uri: Uri? = null
      var resourceId: Int? = null

      if (url.contains("://")) {
        Log.i(TAG, "Parsing URL...")
        uri = Uri.parse(url)
        Log.i(TAG, "Parsed URL: ${uri.toString()}")
      } else {
        Log.i(TAG, "Parsing resourceId...")
        val context = weakContext?.get()
        if (context == null) {
          throw Exception("Context has already been destroyed!")
        }
        resourceId = getResourceId(context, url)
        Log.i(TAG, "Parsed resourceId: $resourceId")
      }

      val bytes = if (uri != null) {
        if (Objects.equals(uri.scheme, "file")) {
          Log.i(TAG, "It's a file URL")
          // It's a file URL
          val path = uri.path ?: throw IOException("File path cannot be null")
          val file = File(path)

          // Check if file exists and is readable
          if (!file.exists() || !file.canRead()) {
            throw IOException("File does not exist or is not readable: $path")
          }

          // Read the file
          FileInputStream(file).use { stream ->
            getLocalFileBytes(stream, file)
          }
        } else {
          Log.i(TAG, "It's a network URL/http resource")
          // It's a network URL/http resource
          val request = Request.Builder().url(uri.toString()).build()
          client.newCall(request).execute().use { response ->
            if (response.isSuccessful && response.body != null) {
              response.body!!.bytes()
            } else {
              throw RuntimeException("Response was not successful!")
            }
          }
        }
      } else if (resourceId != null) {
        Log.i(TAG, "It's bundled into the Android resources/assets")
        // It's bundled into the Android resources/assets
        val context = weakContext?.get()
        if (context == null) {
          throw Exception("Context has already been destroyed!")
        }
        context.resources.openRawResource(resourceId).use { stream ->
          val byteStream = ByteArrayOutputStream()
          val buffer = ByteArray(2048)
          var length: Int
          while (stream.read(buffer).also { length = it } != -1) {
            byteStream.write(buffer, 0, length)
          }
          byteStream.toByteArray()
        }
      } else {
        throw Exception("Input is neither a valid URL, nor a resourceId - cannot load model! (Input: $url)")
      }

      // Convert byte array to ByteBuffer, then wrap with ArrayBuffer
      val byteBuffer = ByteBuffer.allocateDirect(bytes.size)
      byteBuffer.put(bytes)
      byteBuffer.rewind()

      // Create an owning ArrayBuffer by wrapping the ByteBuffer
      return ArrayBuffer.wrap(byteBuffer)
    } catch (e: Exception) {
      Log.e(TAG, "Error fetching byte data: ${e.message}", e)
      throw e
    }
  }
}
