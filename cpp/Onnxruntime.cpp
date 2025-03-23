#include "Onnxruntime.hpp"
#include <NitroModules/Promise.hpp>     // Add this for Promise
#include <NitroModules/ArrayBuffer.hpp> // Already included for ArrayBuffer
#include <stdexcept>
#include <cstring>
#include <NitroModules/NitroLogger.hpp>

// Include NNAPI provider factory for Android
#if defined(__ANDROID__)
#include <nnapi_provider_factory.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#endif

namespace margelo::nitro::nitroonnxruntime
{

  std::string Onnxruntime::getVersion()
  {
    return Ort::GetVersionString();
  }

  void Onnxruntime::configureSessionOptions(Ort::SessionOptions &sessionOptions, const std::optional<SessionOptions> &options)
  {
    if (!options.has_value())
      return;

    // Set thread counts
    if (options->intraOpNumThreads.has_value())
    {
      int intraOpThreads = static_cast<int>(options->intraOpNumThreads.value());
      sessionOptions.SetIntraOpNumThreads(intraOpThreads);
    }

    if (options->interOpNumThreads.has_value())
    {
      int interOpThreads = static_cast<int>(options->interOpNumThreads.value());
      sessionOptions.SetInterOpNumThreads(interOpThreads);
    }

    // Set graph optimization level
    if (options->graphOptimizationLevel.has_value())
    {
      std::string level = options->graphOptimizationLevel.value();
      GraphOptimizationLevel optLevel = ORT_ENABLE_ALL; // default to "all"
      if (level == "disabled")
        optLevel = ORT_DISABLE_ALL;
      else if (level == "basic")
        optLevel = ORT_ENABLE_BASIC;
      else if (level == "extended")
        optLevel = ORT_ENABLE_EXTENDED;
      else if (level == "all")
        optLevel = ORT_ENABLE_ALL;

      sessionOptions.SetGraphOptimizationLevel(optLevel);
    }

    // Set CPU memory arena
    if (options->enableCpuMemArena.has_value())
    {
      bool enable = options->enableCpuMemArena.value();
      if (enable)
        sessionOptions.EnableCpuMemArena();
      else
        sessionOptions.DisableCpuMemArena();
    }

    // Set memory pattern
    if (options->enableMemPattern.has_value())
    {
      bool enable = options->enableMemPattern.value();
      if (enable)
        sessionOptions.EnableMemPattern();
      else
        sessionOptions.DisableMemPattern();
    }

    // Set execution mode
    if (options->executionMode.has_value())
    {
      std::string mode = options->executionMode.value();
      ExecutionMode execMode = ORT_SEQUENTIAL; // default to sequential
      if (mode == "parallel")
        execMode = ORT_PARALLEL;
      sessionOptions.SetExecutionMode(execMode);
    }

    // Set logging ID and severity level
    if (options->logId.has_value())
    {
      sessionOptions.SetLogId(options->logId.value().c_str());
    }

    if (options->logSeverityLevel.has_value())
    {
      int severity = static_cast<int>(options->logSeverityLevel.value());
      sessionOptions.SetLogSeverityLevel(severity);
    }

    // Set execution providers if specified
    if (options->executionProviders.has_value())
    {
      for (const auto &provider : options->executionProviders.value())
      {
        // Simplified provider detection
        if (std::holds_alternative<std::string>(provider))
        {
          std::string providerName = std::get<std::string>(provider);

          if (providerName == "cpu")
          {
            // Using default CPU execution provider
          }
          else if (providerName == "xnnpack")
          {
            sessionOptions.AppendExecutionProvider("XNNPACK", {});
          }
          else if (providerName == "coreml")
          {
            sessionOptions.AppendExecutionProvider("CoreML", {});
          }
          else if (providerName == "nnapi")
          {
            sessionOptions.AppendExecutionProvider("NNAPI", {});
          }
          else
          {
            throw std::runtime_error("Unknown provider name: " + providerName);
          }
        }
        else if (std::holds_alternative<ProviderOptions>(provider))
        {
          ProviderOptions providerOptions = std::get<ProviderOptions>(provider);
          std::string providerName = providerOptions.name;

          if (providerName == "coreml")
          {
#if defined(__APPLE__)
            uint32_t coreml_flags = 0;
            if (providerOptions.useCPUOnly.has_value() && providerOptions.useCPUOnly.value())
              coreml_flags |= COREML_FLAG_USE_CPU_ONLY;
            if (providerOptions.useCPUAndGPU.has_value() && providerOptions.useCPUAndGPU.value())
              coreml_flags |= COREML_FLAG_USE_CPU_AND_GPU;
            if (providerOptions.enableOnSubgraph.has_value() && providerOptions.enableOnSubgraph.value())
              coreml_flags |= COREML_FLAG_ENABLE_ON_SUBGRAPH;
            if (providerOptions.onlyEnableDeviceWithANE.has_value() && providerOptions.onlyEnableDeviceWithANE.value())
              coreml_flags |= COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE;

            OrtSessionOptionsAppendExecutionProvider_CoreML(sessionOptions, coreml_flags);
#else
            throw std::runtime_error("CoreML provider requested but not supported on this platform");
#endif
          }
          else if (providerName == "nnapi")
          {
#if defined(__ANDROID__)
            uint32_t nnapi_flags = 0;
            if (providerOptions.useFP16.has_value() && providerOptions.useFP16.value())
              nnapi_flags |= NNAPI_FLAG_USE_FP16;
            if (providerOptions.useNCHW.has_value() && providerOptions.useNCHW.value())
              nnapi_flags |= NNAPI_FLAG_USE_NCHW;
            if (providerOptions.cpuDisabled.has_value() && providerOptions.cpuDisabled.value())
              nnapi_flags |= NNAPI_FLAG_CPU_DISABLED;
            if (providerOptions.cpuOnly.has_value() && providerOptions.cpuOnly.value())
              nnapi_flags |= NNAPI_FLAG_CPU_ONLY;

            OrtSessionOptionsAppendExecutionProvider_Nnapi(sessionOptions, nnapi_flags);
#else
            throw std::runtime_error("NNAPI provider requested but not supported on this platform");
#endif
          }
          else if (providerName == "xnnpack")
          {
            sessionOptions.AppendExecutionProvider("XNNPACK", {});
          }
          else if (providerName == "cpu")
          {
            // Using CPU execution provider
          }
          else
          {
            throw std::runtime_error("Unknown provider name: " + providerName);
          }
        }
        else
        {
          throw std::runtime_error("Unknown provider type");
        }
      }
    }
  }

  std::shared_ptr<Promise<std::shared_ptr<HybridInferenceSessionSpec>>> Onnxruntime::loadModel(const std::string &modelPath, const std::optional<SessionOptions> &options)
  {
    auto promise = Promise<std::shared_ptr<HybridInferenceSessionSpec>>::create();
    try
    {
      Ort::SessionOptions sessionOptions;
      configureSessionOptions(sessionOptions, options);

      auto session = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions);
      auto inferenceSession = std::make_shared<InferenceSession>(std::move(session));
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

  std::shared_ptr<Promise<std::shared_ptr<HybridInferenceSessionSpec>>> Onnxruntime::loadModelFromBuffer(const std::shared_ptr<ArrayBuffer> &buffer, const std::optional<SessionOptions> &options)
  {
    auto promise = Promise<std::shared_ptr<HybridInferenceSessionSpec>>::create();
    try
    {
      if (!buffer || buffer->size() == 0)
      {
        throw std::runtime_error("Invalid or empty buffer provided for model loading");
      }

      Ort::SessionOptions sessionOptions;
      configureSessionOptions(sessionOptions, options);

      const void *model_data = buffer->data();
      size_t model_size = buffer->size();
      auto session = std::make_unique<Ort::Session>(env_, model_data, model_size, sessionOptions);
      auto inferenceSession = std::make_shared<InferenceSession>(std::move(session));
      promise->resolve(inferenceSession);
    }
    catch (const Ort::Exception &e)
    {
      Logger::log(LogLevel::Error, "Onnxruntime", "Error loading model from buffer: %s", e.what());
      promise->reject(std::make_exception_ptr(e));
    }
    catch (const std::exception &e)
    {
      Logger::log(LogLevel::Error, "Onnxruntime", "Error loading model from buffer: %s", e.what());
      promise->reject(std::make_exception_ptr(e));
    }
    return promise;
  }
} // namespace margelo::nitro::nitroonnxruntime
