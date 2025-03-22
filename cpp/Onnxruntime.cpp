#include "Onnxruntime.hpp"
#include <NitroModules/Promise.hpp>     // Add this for Promise
#include <NitroModules/ArrayBuffer.hpp> // Already included for ArrayBuffer
#include <stdexcept>
#include <cstring>
#include <NitroModules/NitroLogger.hpp>

namespace margelo::nitro::nitroonnxruntime
{

  std::string Onnxruntime::getVersion()
  {
    return Ort::GetVersionString();
  }

  std::shared_ptr<Promise<std::shared_ptr<HybridInferenceSessionSpec>>> Onnxruntime::loadModel(const std::string &modelPath)
  {
    Logger::log(LogLevel::Debug, "Onnxruntime", "Loading model from path: %s", modelPath.c_str());

    auto promise = Promise<std::shared_ptr<HybridInferenceSessionSpec>>::create();
    try
    {
      Ort::SessionOptions sessionOptions;
      auto session = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions);
      auto inferenceSession = std::make_shared<InferenceSession>(std::move(session), modelPath);
      Logger::log(LogLevel::Debug, "Onnxruntime", "Model loaded successfully: %s", modelPath.c_str());
      promise->resolve(inferenceSession);
    }
    catch (const Ort::Exception &e)
    {
      Logger::log(LogLevel::Error, "Onnxruntime", "Error loading model: %s", e.what());
      promise->reject(std::make_exception_ptr(e));
    }
    catch (const std::exception &e)
    {
      Logger::log(LogLevel::Error, "Onnxruntime", "Error loading model: %s", e.what());
      promise->reject(std::make_exception_ptr(e));
    }
    return promise;
  }

  std::shared_ptr<Promise<std::shared_ptr<HybridInferenceSessionSpec>>> Onnxruntime::loadModelFromBuffer(const std::shared_ptr<ArrayBuffer> &buffer)
  {
    auto promise = Promise<std::shared_ptr<HybridInferenceSessionSpec>>::create();
    try
    {
      Ort::SessionOptions sessionOptions;

      // Create a memory object from the buffer
      const void *model_data = buffer->data();
      size_t model_size = buffer->size();

      // Create a session from memory
      auto session = std::make_unique<Ort::Session>(env_, model_data, model_size, sessionOptions);

      // Use a unique identifier for this buffer-loaded model
      std::string key = "buffer_model_" + std::to_string(reinterpret_cast<uintptr_t>(model_data));
      auto inferenceSession = std::make_shared<InferenceSession>(std::move(session), key);

      promise->resolve(inferenceSession);
    }
    catch (const Ort::Exception &e)
    {
      promise->reject(std::make_exception_ptr(e));
    }
    catch (const std::exception &e)
    {
      promise->reject(std::make_exception_ptr(e));
    }
    return promise;
  }

} // namespace margelo::nitro::nitroonnxruntime
