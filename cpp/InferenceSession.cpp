#include "InferenceSession.hpp"
#include <NitroModules/Promise.hpp>
#include <NitroModules/ArrayBuffer.hpp>
#include <stdexcept>
#include <cstring>

namespace margelo::nitro::nitroonnxruntime
{

  void InferenceSession::initializeIONames()
  {
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input names
    size_t numInputs = session_->GetInputCount();
    inputNames_.reserve(numInputs);
    for (size_t i = 0; i < numInputs; i++)
    {
      auto input_name = session_->GetInputNameAllocated(i, allocator);
      inputNames_.push_back(input_name.get());
    }

    // Get output names
    size_t numOutputs = session_->GetOutputCount();
    outputNames_.reserve(numOutputs);
    for (size_t i = 0; i < numOutputs; i++)
    {
      auto output_name = session_->GetOutputNameAllocated(i, allocator);
      outputNames_.push_back(output_name.get());
    }
  }

  std::string InferenceSession::getKey()
  {
    return key_;
  }

  std::vector<std::string> InferenceSession::getInputNames()
  {
    return inputNames_;
  }

  std::vector<std::string> InferenceSession::getOutputNames()
  {
    return outputNames_;
  }

  std::shared_ptr<Promise<std::unordered_map<std::string, std::shared_ptr<ArrayBuffer>>>> InferenceSession::run(
      const std::unordered_map<std::string, EncodedTensor> &feeds)
  {
    auto promise = Promise<std::unordered_map<std::string, std::shared_ptr<ArrayBuffer>>>::create();

    try
    {
      Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      std::vector<const char *> inputNames;
      std::vector<Ort::Value> inputTensors;

      // Prepare inputs
      for (const auto &[name, tensor] : feeds)
      {
        inputNames.push_back(name.c_str());

        const void *data = tensor.data->data();
        size_t byteSize = tensor.data->size();

        // Create tensor
        if (tensor.type == "float32")
        {
          // Convert dims from double to int64_t
          std::vector<int64_t> dims_int64(tensor.dims.size());
          for (size_t i = 0; i < tensor.dims.size(); i++)
          {
            dims_int64[i] = static_cast<int64_t>(tensor.dims[i]);
          }

          inputTensors.emplace_back(Ort::Value::CreateTensor<float>(
              memoryInfo,
              reinterpret_cast<float *>(const_cast<void *>(data)),
              byteSize / sizeof(float),
              dims_int64.data(),
              dims_int64.size()));
        }
        else
        {
          throw std::runtime_error("Unsupported tensor type: " + tensor.type);
        }
      }

      // Prepare output names
      std::vector<const char *> outputNamesC;
      for (const auto &name : outputNames_)
      {
        outputNamesC.push_back(name.c_str());
      }

      // Run inference
      auto outputTensors = session_->Run(Ort::RunOptions{nullptr},
                                         inputNames.data(), inputTensors.data(), inputTensors.size(),
                                         outputNamesC.data(), outputNamesC.size());

      // Process output
      std::unordered_map<std::string, std::shared_ptr<ArrayBuffer>> results;
      for (size_t i = 0; i < outputTensors.size(); ++i)
      {
        // For simplicity, assuming outputs are float tensors
        float *outputData = outputTensors[i].GetTensorMutableData<float>();
        auto tensorInfo = outputTensors[i].GetTensorTypeAndShapeInfo();
        size_t floatCount = tensorInfo.GetElementCount();
        size_t byteSize = floatCount * sizeof(float);

        // Create ArrayBuffer with the output data
        float *buffer = new float[floatCount];
        std::memcpy(buffer, outputData, byteSize);
        // Use ArrayBuffer::wrap to create an owning buffer
        auto arrayBuffer = ArrayBuffer::wrap(
            reinterpret_cast<uint8_t *>(buffer), byteSize, [buffer]()
            { delete[] buffer; });

        results[outputNames_[i]] = arrayBuffer;
      }

      promise->resolve(results);
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

  std::shared_ptr<Promise<void>> InferenceSession::close()
  {
    auto promise = Promise<void>::create();

    try
    {
      session_.reset();
      promise->resolve();
    }
    catch (const std::exception &e)
    {
      promise->reject(std::make_exception_ptr(e));
    }

    return promise;
  }

} // namespace margelo::nitro::nitroonnxruntime
