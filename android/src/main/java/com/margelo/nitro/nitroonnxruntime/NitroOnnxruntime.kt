package com.margelo.nitro.nitroonnxruntime
  
import com.facebook.proguard.annotations.DoNotStrip

@DoNotStrip
class NitroOnnxruntime : HybridNitroOnnxruntimeSpec() {
  override fun multiply(a: Double, b: Double): Double {
    return a * b
  }
}
