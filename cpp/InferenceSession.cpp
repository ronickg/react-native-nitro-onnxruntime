#include "InferenceSession.hpp"
#include <NitroModules/Promise.hpp>
#include <NitroModules/ArrayBuffer.hpp>
#include <stdexcept>
#include <cstring>

namespace margelo::nitro::nitroonnxruntime
{
  std::string getTypeString(ONNXTensorElementDataType type)
  {
    switch (type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return "int8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return "int16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return "bool";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return "float64";
    default:
      throw std::runtime_error("Unsupported tensor type: " + std::to_string(type));
    }
  }

  void InferenceSession::initializeIONames()
  {
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input names
    size_t numInputs = session_->GetInputCount();
    inputNames_.reserve(numInputs);
    for (size_t i = 0; i < numInputs; i++)
    {
      auto input_name = session_->GetInputNameAllocated(i, allocator);
      auto info = session_->GetInputTypeInfo(i);
      auto tensorInfo = info.GetTensorTypeAndShapeInfo();
      auto dims = tensorInfo.GetShape();
      std::vector<double> dims_double(dims.begin(), dims.end());

      // Get actual type from model
      std::string type = getTypeString(tensorInfo.GetElementType());

      inputNames_.emplace_back(dims_double, type, input_name.get());
    }

    // Get output names
    size_t numOutputs = session_->GetOutputCount();
    outputNames_.reserve(numOutputs);
    for (size_t i = 0; i < numOutputs; i++)
    {
      auto output_name = session_->GetOutputNameAllocated(i, allocator);
      auto info = session_->GetOutputTypeInfo(i);
      auto tensorInfo = info.GetTensorTypeAndShapeInfo();
      auto dims = tensorInfo.GetShape();
      std::vector<double> dims_double(dims.begin(), dims.end());

      // Get actual type from model
      std::string type = getTypeString(tensorInfo.GetElementType());

      outputNames_.emplace_back(dims_double, type, output_name.get());
    }
  }

  template <typename T>
  Ort::Value createTensor(Ort::MemoryInfo &memoryInfo, const void *data, size_t byteSize,
                          const std::vector<int64_t> &dims)
  {
    T *buffer = new T[byteSize / sizeof(T)];
    std::memcpy(buffer, data, byteSize);
    return Ort::Value::CreateTensor<T>(
        memoryInfo,
        buffer,
        byteSize / sizeof(T),
        dims.data(),
        dims.size());
  }

  std::string InferenceSession::getKey()
  {
    return key_;
  }

  std::vector<Tensor> InferenceSession::getInputNames()
  {
    return inputNames_;
  }

  std::vector<Tensor> InferenceSession::getOutputNames()
  {
    return outputNames_;
  }

  std::shared_ptr<Promise<std::unordered_map<std::string, std::shared_ptr<ArrayBuffer>>>> InferenceSession::run(
      const std::unordered_map<std::string, std::shared_ptr<ArrayBuffer>> &feeds)
  {
    auto promise = Promise<std::unordered_map<std::string, std::shared_ptr<ArrayBuffer>>>::create();

    try
    {
      Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      std::vector<const char *> inputNames;
      std::vector<Ort::Value> inputTensors;

      // Prepare inputs
      for (const auto &[name, buffer] : feeds)
      {
        inputNames.push_back(name.c_str());

        // Since this is a non-owning buffer from JS, we need to access it safely within the sync call
        const void *data = buffer->data();
        size_t byteSize = buffer->size();

        // Find corresponding input tensor info
        auto it = std::find_if(inputNames_.begin(), inputNames_.end(),
                               [&name](const Tensor &t)
                               { return t.name == name; });
        if (it == inputNames_.end())
        {
          throw std::runtime_error("Input name not found: " + name);
        }

        // Convert dims from double to int64_t
        std::vector<int64_t> dims_int64(it->dims.size());
        for (size_t i = 0; i < it->dims.size(); i++)
        {
          dims_int64[i] = static_cast<int64_t>(it->dims[i]);
        }

        // Create tensor based on type
        if (it->type == "float32")
        {
          inputTensors.push_back(createTensor<float>(memoryInfo, data, byteSize, dims_int64));
        }
        else if (it->type == "int8" || it->type == "bool")
        {
          inputTensors.push_back(createTensor<int8_t>(memoryInfo, data, byteSize, dims_int64));
        }
        else if (it->type == "uint8")
        {
          inputTensors.push_back(createTensor<uint8_t>(memoryInfo, data, byteSize, dims_int64));
        }
        else if (it->type == "int16")
        {
          inputTensors.push_back(createTensor<int16_t>(memoryInfo, data, byteSize, dims_int64));
        }
        else if (it->type == "int32")
        {
          inputTensors.push_back(createTensor<int32_t>(memoryInfo, data, byteSize, dims_int64));
        }
        else if (it->type == "int64")
        {
          inputTensors.push_back(createTensor<int64_t>(memoryInfo, data, byteSize, dims_int64));
        }
        else if (it->type == "float64")
        {
          inputTensors.push_back(createTensor<double>(memoryInfo, data, byteSize, dims_int64));
        }
        else
        {
          throw std::runtime_error("Unsupported tensor type: " + it->type);
        }
      }

      // Prepare output names
      std::vector<const char *> outputNamesC;
      for (const auto &tensor : outputNames_)
      {
        outputNamesC.push_back(tensor.name.c_str());
      }

      // Run inference
      auto outputTensors = session_->Run(Ort::RunOptions{nullptr},
                                         inputNames.data(), inputTensors.data(), inputTensors.size(),
                                         outputNamesC.data(), outputNamesC.size());

      // Process output
      std::unordered_map<std::string, std::shared_ptr<ArrayBuffer>> results;
      for (size_t i = 0; i < outputTensors.size(); ++i)
      {
        auto tensorInfo = outputTensors[i].GetTensorTypeAndShapeInfo();
        size_t elementCount = tensorInfo.GetElementCount();
        size_t elementSize;
        const void *outputData;

        // Get the correct data pointer and element size based on type
        if (outputNames_[i].type == "float32")
        {
          outputData = outputTensors[i].GetTensorMutableData<float>();
          elementSize = sizeof(float);
        }
        else if (outputNames_[i].type == "int8" || outputNames_[i].type == "bool")
        {
          outputData = outputTensors[i].GetTensorMutableData<int8_t>();
          elementSize = sizeof(int8_t);
        }
        else if (outputNames_[i].type == "uint8")
        {
          outputData = outputTensors[i].GetTensorMutableData<uint8_t>();
          elementSize = sizeof(uint8_t);
        }
        else if (outputNames_[i].type == "int16")
        {
          outputData = outputTensors[i].GetTensorMutableData<int16_t>();
          elementSize = sizeof(int16_t);
        }
        else if (outputNames_[i].type == "int32")
        {
          outputData = outputTensors[i].GetTensorMutableData<int32_t>();
          elementSize = sizeof(int32_t);
        }
        else if (outputNames_[i].type == "int64")
        {
          outputData = outputTensors[i].GetTensorMutableData<int64_t>();
          elementSize = sizeof(int64_t);
        }
        else if (outputNames_[i].type == "float64")
        {
          outputData = outputTensors[i].GetTensorMutableData<double>();
          elementSize = sizeof(double);
        }
        else
        {
          throw std::runtime_error("Unsupported output tensor type: " + outputNames_[i].type);
        }

        size_t byteSize = elementCount * elementSize;

        // Create a new owning buffer for the output data
        auto buffer = ArrayBuffer::allocate(byteSize);
        std::memcpy(buffer->data(), outputData, byteSize);

        results.emplace(outputNames_[i].name, buffer);
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
