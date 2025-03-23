#pragma once

#include "HybridInferenceSessionSpec.hpp"
#include "onnxruntime_cxx_api.h"
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
    InferenceSession(std::unique_ptr<Ort::Session> session)
        : HybridObject(TAG), session_(std::move(session))
    {
      initializeIONames();
    }

    // Destructor - will be called automatically when shared_ptr ref count hits 0
    ~InferenceSession() override
    {
      // Call dispose to ensure cleanup
      dispose();
    }

  public:
    // Implementation of pure virtual methods from spec
    std::vector<Tensor> getInputNames() override;
    std::vector<Tensor> getOutputNames() override;
    std::shared_ptr<Promise<std::unordered_map<std::string, std::shared_ptr<ArrayBuffer>>>> run(
        const std::unordered_map<std::string, std::shared_ptr<ArrayBuffer>> &feeds) override;
    void dispose() override;

  private:
    std::unique_ptr<Ort::Session> session_;
    std::vector<Tensor> inputNames_;
    std::vector<Tensor> outputNames_;

    void initializeIONames();
  };

} // namespace margelo::nitro::nitroonnxruntime
