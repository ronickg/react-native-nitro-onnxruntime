#pragma once

#include "HybridOnnxruntimeSpec.hpp"
#include "InferenceSession.hpp"
#include "onnxruntime_cxx_api.h"
#include <memory>
#include <unordered_map>

// Include provider headers based on platform
#if defined(__APPLE__)
#include "coreml_provider_factory.h"
#endif

#if defined(__ANDROID__)
#include "nnapi_provider_factory.h"
#endif

namespace margelo::nitro::nitroonnxruntime
{

  class Onnxruntime : public virtual HybridOnnxruntimeSpec
  {
  public:
    // Constructor
    Onnxruntime() : HybridObject(TAG), env_(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Onnxruntime")) {}

    // Destructor
    ~Onnxruntime() override = default;

  public:
    // Implementation of pure virtual methods from spec
    std::string getVersion() override;
    std::shared_ptr<Promise<std::shared_ptr<HybridInferenceSessionSpec>>> loadModel(const std::string &modelPath, const std::optional<SessionOptions> &options = std::nullopt) override;
    std::shared_ptr<Promise<std::shared_ptr<HybridInferenceSessionSpec>>> loadModelFromBuffer(const std::shared_ptr<ArrayBuffer> &buffer, const std::optional<SessionOptions> &options = std::nullopt) override;

  private:
    void configureSessionOptions(Ort::SessionOptions &sessionOptions, const std::optional<SessionOptions> &options);
    // ONNX Runtime environment (shared across sessions)
    Ort::Env env_;
  };

} // namespace margelo::nitro::nitroonnxruntime
