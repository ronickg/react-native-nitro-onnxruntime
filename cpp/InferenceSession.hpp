#pragma once

#include "HybridInferenceSessionSpec.hpp"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace margelo::nitro::nitroonnxruntime
{

  class InferenceSession : public virtual HybridInferenceSessionSpec
  {
  public:
    InferenceSession() : HybridObject(TAG) {}
    // Constructor
    InferenceSession(std::unique_ptr<Ort::Session> session, std::string key)
        : HybridObject(TAG),
          session_(std::move(session)),
          key_(std::move(key))
    {
      initializeIONames();
    }

    // Destructor
    ~InferenceSession() override = default;

  public:
    // Implementation of pure virtual methods from spec
    std::string getKey() override;
    std::vector<std::string> getInputNames() override;
    std::vector<std::string> getOutputNames() override;
    std::shared_ptr<Promise<std::unordered_map<std::string, std::shared_ptr<ArrayBuffer>>>> run(
        const std::unordered_map<std::string, EncodedTensor> &feeds) override;
    std::shared_ptr<Promise<void>> close() override;

  private:
    std::unique_ptr<Ort::Session> session_;
    std::string key_;
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;

    void initializeIONames();
  };

} // namespace margelo::nitro::nitroonnxruntime
