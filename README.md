# react-native-nitro-onnxruntime

High-performance ONNX Runtime bindings for React Native, built with [Nitro Modules](https://nitro.margelo.com/) for maximum performance.

## Features

- ⚡ **Blazing Fast**: Built with Nitro Modules for zero-overhead JSI bindings
- 🎯 **Hardware Acceleration**: Support for NNAPI (Android), CoreML (iOS), and XNNPACK
- 🔄 **Modern API**: Promise-based async API with TypeScript support
- 📦 **Flexible Model Loading**: Load models from files, URLs, or buffers
- 🎨 **Full Type Support**: Complete TypeScript definitions
- 🔧 **Configurable**: Extensive session options for optimization

## Installation

```sh
npm install react-native-nitro-onnxruntime react-native-nitro-modules
```

> **Note**: `react-native-nitro-modules` is required as this library relies on [Nitro Modules](https://nitro.margelo.com/).

## Usage

### Basic Example

```typescript
import ort from 'react-native-nitro-onnxruntime';

// Load a model
const session = await ort.loadModel('path/to/model.onnx');

// Get input/output information
console.log('Inputs:', session.inputNames);
console.log('Outputs:', session.outputNames);

// Prepare input data
const inputData = new Float32Array(1 * 3 * 224 * 224); // Batch=1, Channels=3, Height=224, Width=224
// ... fill inputData with your data

// Run inference
const results = await session.run({
  [session.inputNames[0].name]: inputData.buffer
});

// Access output
const outputBuffer = results[session.outputNames[0].name];
const outputData = new Float32Array(outputBuffer);
console.log('Output:', outputData);
```

### Loading Models from Assets

Models can be loaded using `require()` for bundled assets:

```typescript
const session = await ort.loadModel(require('./assets/model.onnx'));
```

This automatically copies the model to the device's file system on first load.

### Hardware Acceleration

#### Android (NNAPI)

```typescript
const session = await ort.loadModel('model.onnx', {
  executionProviders: ['nnapi']
});

// Or with options
const session = await ort.loadModel('model.onnx', {
  executionProviders: [{
    name: 'nnapi',
    useFP16: true,        // Use FP16 precision
    cpuDisabled: true,    // Disable CPU fallback
  }]
});
```

#### iOS (CoreML)

```typescript
const session = await ort.loadModel('model.onnx', {
  executionProviders: ['coreml']
});

// Or with options
const session = await ort.loadModel('model.onnx', {
  executionProviders: [{
    name: 'coreml',
    useCPUOnly: false,
    onlyEnableDeviceWithANE: true,  // Only use devices with Apple Neural Engine
  }]
});
```

#### XNNPACK (Cross-platform)

```typescript
const session = await ort.loadModel('model.onnx', {
  executionProviders: ['xnnpack']
});
```

### Advanced Configuration

```typescript
const session = await ort.loadModel('model.onnx', {
  // Thread configuration
  intraOpNumThreads: 4,
  interOpNumThreads: 2,
  
  // Graph optimization
  graphOptimizationLevel: 'all', // 'disabled' | 'basic' | 'extended' | 'all'
  
  // Memory settings
  enableCpuMemArena: true,
  enableMemPattern: true,
  
  // Execution mode
  executionMode: 'sequential', // 'sequential' | 'parallel'
  
  // Logging
  logId: 'MyModel',
  logSeverityLevel: 2, // 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal
  
  // Execution providers
  executionProviders: ['nnapi', 'cpu']
});
```

### Loading from Buffer

```typescript
import RNFS from 'react-native-fs';

// Load model file into buffer
const modelPath = 'path/to/model.onnx';
const modelBuffer = await RNFS.readFile(modelPath, 'base64');
const arrayBuffer = Uint8Array.from(atob(modelBuffer), c => c.charCodeAt(0)).buffer;

// Create session from buffer
const session = await ort.loadModelFromBuffer(arrayBuffer, {
  executionProviders: ['nnapi']
});
```

### Memory Management

```typescript
// Dispose of session when done
session.dispose();
```

## API Reference

### `ort.getVersion()`

Returns the ONNX Runtime version string.

```typescript
const version = ort.getVersion();
console.log('ONNX Runtime version:', version);
```

### `ort.loadModel(path, options?)`

Load an ONNX model from a file path or `require()` asset.

**Parameters:**
- `path`: `string` | `number` (from `require()`) - Path to the model file
- `options`: `SessionOptions` (optional) - Configuration options

**Returns:** `Promise<InferenceSession>`

### `ort.loadModelFromBuffer(buffer, options?)`

Load an ONNX model from an ArrayBuffer.

**Parameters:**
- `buffer`: `ArrayBuffer` - Model data
- `options`: `SessionOptions` (optional) - Configuration options

**Returns:** `Promise<InferenceSession>`

### `InferenceSession`

#### `session.inputNames`

Array of input tensor information:
```typescript
type Tensor = {
  name: string;
  dims: number[];  // Shape, negative values indicate dynamic dimensions
  type: string;    // 'float32', 'int64', etc.
};
```

#### `session.outputNames`

Array of output tensor information (same format as `inputNames`).

#### `session.run(feeds)`

Run inference with the given inputs.

**Parameters:**
- `feeds`: `Record<string, ArrayBuffer>` - Map of input names to ArrayBuffers

**Returns:** `Promise<Record<string, ArrayBuffer>>` - Map of output names to ArrayBuffers

#### `session.dispose()`

Free the session and release resources.

### `SessionOptions`

```typescript
type SessionOptions = {
  intraOpNumThreads?: number;
  interOpNumThreads?: number;
  graphOptimizationLevel?: 'disabled' | 'basic' | 'extended' | 'all';
  enableCpuMemArena?: boolean;
  enableMemPattern?: boolean;
  executionMode?: 'sequential' | 'parallel';
  logId?: string;
  logSeverityLevel?: number;
  executionProviders?: (string | ProviderOptions)[];
};

type ProviderOptions = {
  name: 'nnapi' | 'coreml' | 'xnnpack';
  // NNAPI options (Android)
  useFP16?: boolean;
  useNCHW?: boolean;
  cpuDisabled?: boolean;
  cpuOnly?: boolean;
  // CoreML options (iOS)
  useCPUOnly?: boolean;
  useCPUAndGPU?: boolean;
  enableOnSubgraph?: boolean;
  onlyEnableDeviceWithANE?: boolean;
};
```

## Supported Platforms

- ✅ Android (API 21+)
- ✅ iOS (13.0+)

## Supported Data Types

- `float32` (Float)
- `float64` (Double)
- `int8`
- `uint8`
- `int16`
- `int32`
- `int64`
- `bool`

## Performance Tips

1. **Use Hardware Acceleration**: Enable NNAPI (Android) or CoreML (iOS) for better performance
2. **Optimize Thread Count**: Set `intraOpNumThreads` based on your device's CPU cores
3. **Enable Graph Optimization**: Use `graphOptimizationLevel: 'all'` for production
4. **Reuse Sessions**: Create the session once and reuse it for multiple inferences
5. **Use FP16**: Enable `useFP16` on NNAPI for faster inference with acceptable accuracy loss

## Troubleshooting

### Android Build Issues

If you encounter duplicate library errors, ensure your `android/app/build.gradle` has:

```gradle
android {
    packaging {
        jniLibs {
            pickFirsts += ['**/libonnxruntime.so']
        }
    }
}
```

### Memory Issues

Always call `session.dispose()` when you're done with a session to free up memory.

## Example App

See the [example](./example) directory for a complete working example with speed comparisons.

## Contributing

See the [contributing guide](CONTRIBUTING.md) to learn how to contribute to the repository and the development workflow.

## License

MIT

---

Made with [Nitro Modules](https://nitro.margelo.com/) and [create-react-native-library](https://github.com/callstack/react-native-builder-bob)
