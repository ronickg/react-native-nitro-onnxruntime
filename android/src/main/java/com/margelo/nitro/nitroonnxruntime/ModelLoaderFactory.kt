package com.margelo.nitro.nitroonnxruntime

import android.annotation.SuppressLint
import androidx.annotation.Keep
import com.facebook.proguard.annotations.DoNotStrip
import com.margelo.nitro.core.Promise
import com.margelo.nitro.NitroModules
import java.io.File
import java.io.FileOutputStream
import okhttp3.OkHttpClient
import okhttp3.Request

@Keep
@DoNotStrip
class ModelLoaderFactory : HybridModelLoaderFactorySpec() {
  companion object {
    private val client = OkHttpClient()
  }

  // region Private Helpers

  private fun modelsDirectory(): File {
    val context = NitroModules.applicationContext
        ?: throw Error("Application context is unavailable")
    val dir = File(context.getExternalFilesDir(null), "onnx_models")
    dir.mkdirs()
    return dir
  }

  private fun extractFileName(source: String): String {
    val clean = if (source.contains("?")) source.substring(0, source.indexOf("?")) else source
    return File(clean).name
  }

  // endregion

  // region HybridModelLoaderFactorySpec

  override fun createFileModelLoader(filePath: String): Promise<String> {
    return Promise.async {
      val cleanPath = if (filePath.startsWith("file://")) filePath.removePrefix("file://") else filePath
      val file = File(cleanPath)
      if (!file.exists()) {
        throw Error("Model file not found at path: $cleanPath")
      }
      cleanPath
    }
  }

  @SuppressLint("DiscouragedApi")
  override fun createResourceModelLoader(name: String): Promise<String> {
    return Promise.async {
      val context = NitroModules.applicationContext
          ?: throw Error("Application context is unavailable")

      val fileName = extractFileName(name)
      val destination = File(modelsDirectory(), fileName)

      if (destination.exists()) {
        return@async destination.absolutePath
      }

      // Try drawable resources first (for require() assets bundled into res/raw in release)
      val rawResourceId = context.resources.getIdentifier(name, "drawable", context.packageName)
      if (rawResourceId != 0) {
        context.resources.openRawResource(rawResourceId).use { input ->
          FileOutputStream(destination).use { output ->
            input.copyTo(output)
          }
        }
        return@async destination.absolutePath
      }

      // Fall back to assets
      context.assets.open(name).use { input ->
        FileOutputStream(destination).use { output ->
          input.copyTo(output)
        }
      }

      destination.absolutePath
    }
  }

  override fun createUrlModelLoader(url: String): Promise<String> {
    return Promise.async {
      val fileName = extractFileName(url)
      val destination = File(modelsDirectory(), fileName)

      // Return cached file if it exists
      if (destination.exists()) {
        return@async destination.absolutePath
      }

      val request = Request.Builder().url(url).build()
      client.newCall(request).execute().use { response ->
        if (!response.isSuccessful) {
          throw Error("HTTP error ${response.code} downloading from $url")
        }
        response.body?.byteStream()?.use { input ->
          FileOutputStream(destination).use { output ->
            input.copyTo(output)
          }
        } ?: throw Error("Empty response body from $url")
      }

      destination.absolutePath
    }
  }

  // endregion
}
