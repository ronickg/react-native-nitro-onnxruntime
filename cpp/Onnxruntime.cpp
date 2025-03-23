#include "Onnxruntime.hpp"
#include <NitroModules/Promise.hpp>     // Add this for Promise
#include <NitroModules/ArrayBuffer.hpp> // Already included for ArrayBuffer
#include <stdexcept>
#include <cstring>
#include <NitroModules/NitroLogger.hpp>

// Include NNAPI provider factory for Android
#if defined(__ANDROID__)
#include <nnapi_provider_factory.h>
#endif

namespace margelo::nitro::nitroonnxruntime
{

  std::string Onnxruntime::getVersion()
  {
    return Ort::GetVersionString();
  }

  std::shared_ptr<Promise<std::shared_ptr<HybridInferenceSessionSpec>>> Onnxruntime::loadModel(const std::string &modelPath, const std::optional<SessionOptions> &options)
  {
    Logger::log(LogLevel::Debug, "Onnxruntime", "Loading model from path: %s", modelPath.c_str());

    auto promise = Promise<std::shared_ptr<HybridInferenceSessionSpec>>::create();
    try
    {
      Ort::SessionOptions sessionOptions;

      // Configure session options if provided
      if (options.has_value())
      {
        // Set execution providers if specified
        if (options->executionProviders.has_value())
        {
          for (const auto &provider : options->executionProviders.value())
          {
            if (provider == "cpu")
            {
              // CPU provider is enabled by default
            }
            else if (provider == "nnapi")
            {
// Enable NNAPI provider if available
#if defined(__ANDROID__)
              // Use NNAPI provider with default flags
              OrtSessionOptionsAppendExecutionProvider_Nnapi(sessionOptions, 0);
#endif
            }
            // Add other providers as needed
          }
        }

        // Set optimization level using config entry
        if (options->optimizationLevel.has_value())
        {
          sessionOptions.AddConfigEntry("session.graph_optimization_level",
                                        std::to_string(static_cast<int>(options->optimizationLevel.value())).c_str());
        }

        // Set memory pattern
        if (options->enableMemoryPattern.has_value())
        {
          if (options->enableMemoryPattern.value())
          {
            sessionOptions.EnableMemPattern();
          }
          else
          {
            sessionOptions.DisableMemPattern();
          }
        }

        // Set thread counts
        if (options->intraOpNumThreads.has_value())
        {
          sessionOptions.SetIntraOpNumThreads(static_cast<int>(options->intraOpNumThreads.value()));
        }

        if (options->interOpNumThreads.has_value())
        {
          sessionOptions.SetInterOpNumThreads(static_cast<int>(options->interOpNumThreads.value()));
        }

        // Set graph optimization level
        if (options->graphOptimizationLevel.has_value())
        {
          int level = static_cast<int>(options->graphOptimizationLevel.value());
          // GraphOptimizationLevel is exposed via an enum, convert the integer value
          // 0=disable, 1=basic, 2=extended, 3=all
          sessionOptions.SetGraphOptimizationLevel(static_cast<GraphOptimizationLevel>(level));
        }

        // Set logging level
        if (options->logSeverityLevel.has_value())
        {
          sessionOptions.SetLogSeverityLevel(static_cast<int>(options->logSeverityLevel.value()));
        }

        // Set execution mode
        if (options->executionMode.has_value())
        {
          ExecutionMode mode = static_cast<int>(options->executionMode.value()) == 0 ? ExecutionMode::ORT_SEQUENTIAL : ExecutionMode::ORT_PARALLEL;
          sessionOptions.SetExecutionMode(mode);
        }
      }

      auto session = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions);
      auto inferenceSession = std::make_shared<InferenceSession>(std::move(session));
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

  std::shared_ptr<Promise<std::shared_ptr<HybridInferenceSessionSpec>>> Onnxruntime::loadModelFromBuffer(const std::shared_ptr<ArrayBuffer> &buffer, const std::optional<SessionOptions> &options)
  {
    auto promise = Promise<std::shared_ptr<HybridInferenceSessionSpec>>::create();
    try
    {
      Ort::SessionOptions sessionOptions;

      // Configure session options if provided
      if (options.has_value())
      {
        // Set execution providers if specified
        if (options->executionProviders.has_value())
        {
          for (const auto &provider : options->executionProviders.value())
          {
            if (provider == "cpu")
            {
              // CPU provider is enabled by default
            }
            else if (provider == "nnapi")
            {
// Enable NNAPI provider if available
#if defined(__ANDROID__)
              // Use NNAPI provider with default flags
              OrtSessionOptionsAppendExecutionProvider_Nnapi(sessionOptions, 0);
#endif
            }
            // Add other providers as needed
          }
        }

        // Set optimization level using config entry
        if (options->optimizationLevel.has_value())
        {
          sessionOptions.AddConfigEntry("session.graph_optimization_level",
                                        std::to_string(static_cast<int>(options->optimizationLevel.value())).c_str());
        }

        // Set memory pattern
        if (options->enableMemoryPattern.has_value())
        {
          if (options->enableMemoryPattern.value())
          {
            sessionOptions.EnableMemPattern();
          }
          else
          {
            sessionOptions.DisableMemPattern();
          }
        }

        // Set thread counts
        if (options->intraOpNumThreads.has_value())
        {
          sessionOptions.SetIntraOpNumThreads(static_cast<int>(options->intraOpNumThreads.value()));
        }

        if (options->interOpNumThreads.has_value())
        {
          sessionOptions.SetInterOpNumThreads(static_cast<int>(options->interOpNumThreads.value()));
        }

        // Set graph optimization level
        if (options->graphOptimizationLevel.has_value())
        {
          int level = static_cast<int>(options->graphOptimizationLevel.value());
          // GraphOptimizationLevel is exposed via an enum, convert the integer value
          // 0=disable, 1=basic, 2=extended, 3=all
          sessionOptions.SetGraphOptimizationLevel(static_cast<GraphOptimizationLevel>(level));
        }

        // Set logging level
        if (options->logSeverityLevel.has_value())
        {
          sessionOptions.SetLogSeverityLevel(static_cast<int>(options->logSeverityLevel.value()));
        }

        // Set execution mode
        if (options->executionMode.has_value())
        {
          ExecutionMode mode = static_cast<int>(options->executionMode.value()) == 0 ? ExecutionMode::ORT_SEQUENTIAL : ExecutionMode::ORT_PARALLEL;
          sessionOptions.SetExecutionMode(mode);
        }
      }

      // Create a memory object from the buffer
      const void *model_data = buffer->data();
      size_t model_size = buffer->size();

      // Create a session from memory
      auto session = std::make_unique<Ort::Session>(env_, model_data, model_size, sessionOptions);

      // Use a unique identifier for this buffer-loaded model
      auto inferenceSession = std::make_shared<InferenceSession>(std::move(session));

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
