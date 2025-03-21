#include <jni.h>
#include "nitroonnxruntimeOnLoad.hpp"

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return margelo::nitro::nitroonnxruntime::initialize(vm);
}
