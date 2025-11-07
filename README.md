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

Models can be loaded using `require()` for bundled assets. The library automatically copies the model from your app bundle to the device's file system and caches it:

```typescript
// Load from bundled asset
const session = await ort.loadModel(require('./assets/model.onnx'));
```

You can also load from a URL:

```typescript
// Load from URL
const session = await ort.loadModel({
  url: 'http://example.com/model.onnx'
});
```

Or from a file path:

```typescript
// Load from file path
const session = await ort.loadModel('/path/to/model.onnx');
```

**Note:** When using `require()` or `{ url }`, the model is automatically copied to the app's files directory and cached. Subsequent loads will use the cached file for faster initialization.

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

For advanced use cases, you can load models directly from an ArrayBuffer:

```typescript
import RNFS from 'react-native-fs';

// Option 1: Load from file system
const modelPath = 'path/to/model.onnx';
const base64Data = await RNFS.readFile(modelPath, 'base64');
const arrayBuffer = Uint8Array.from(atob(base64Data), c => c.charCodeAt(0)).buffer;

const session = await ort.loadModelFromBuffer(arrayBuffer, {
  executionProviders: ['nnapi']
});

// Option 2: Load from network
const response = await fetch('https://example.com/model.onnx');
const arrayBuffer = await response.arrayBuffer();

const session = await ort.loadModelFromBuffer(arrayBuffer, {
  executionProviders: ['coreml']
});
```

**Use cases for `loadModelFromBuffer`:**
- Loading models from encrypted storage
- Downloading models from authenticated endpoints
- Processing models before loading (e.g., decompression)
- Dynamic model generation

### React Hooks

The library provides a convenient React hook for loading models:

```typescript
import { useLoadModel } from 'react-native-nitro-onnxruntime';

function MyComponent() {
  const modelState = useLoadModel(require('./assets/model.onnx'), {
    executionProviders: ['nnapi']
  });

  if (modelState.state === 'loading') {
    return <Text>Loading model...</Text>;
  }

  if (modelState.state === 'error') {
    return <Text>Error: {modelState.error.message}</Text>;
  }

  // modelState.state === 'loaded'
  const session = modelState.model;
  
  // Use session for inference
  const runInference = async () => {
    const input = new Float32Array(224 * 224 * 3);
    const results = await session.run({
      [session.inputNames[0].name]: input.buffer
    });
  };

  return <Button onPress={runInference} title="Run Inference" />;
}
```

### Memory Management

Sessions are automatically cleaned up by Nitro Modules when they go out of scope. However, you can manually dispose of a session to free memory immediately:

```typescript
// Optional: Dispose of session early to free memory immediately
session.dispose();
```

**When to call `dispose()`:**
- Loading multiple models and want to free memory between loads
- Memory-constrained environments
- Want immediate cleanup instead of waiting for garbage collection

**Note:** You don't need to call `dispose()` in most cases - Nitro Modules will automatically clean up when the session is no longer referenced.

## API Reference

### `ort.getVersion()`

Returns the ONNX Runtime version string.

```typescript
const version = ort.getVersion();
console.log('ONNX Runtime version:', version);
```

### `ort.loadModel(source, options?)`

Load an ONNX model from various sources.

**Parameters:**
- `source`: `string` | `number` | `{ url: string }` - Model source:
  - `string`: File path on device
  - `number`: `require()` asset (automatically copied to files directory)
  - `{ url: string }`: URL to download from (automatically cached)
- `options`: `SessionOptions` (optional) - Configuration options

**Returns:** `Promise<InferenceSession>`

**Example:**
```typescript
// From bundled asset
const session1 = await ort.loadModel(require('./model.onnx'));

// From file path
const session2 = await ort.loadModel('/data/user/model.onnx');

// From URL
const session3 = await ort.loadModel({ url: 'https://example.com/model.onnx' });
```

### `ort.loadModelFromBuffer(buffer, options?)`

Load an ONNX model from an ArrayBuffer.

**Parameters:**
- `buffer`: `ArrayBuffer` - Model data
- `options`: `SessionOptions` (optional) - Configuration options

**Returns:** `Promise<InferenceSession>`

### `copyFile(source)`

Manually copy a model file from a bundled asset or URL to the device's file system. This is useful if you want to copy the file before loading it.

**Parameters:**
- `source`: `number` | `{ url: string }` - Model source to copy

**Returns:** `Promise<string>` - Path to the copied file

**Example:**
```typescript
// Copy bundled asset
const modelPath = await copyFile(require('./model.onnx'));
console.log('Model copied to:', modelPath);

// Now you can load it
const session = await ort.loadModel(modelPath);

// Or copy from URL
const urlPath = await copyFile({ url: 'https://example.com/model.onnx' });
const session2 = await ort.loadModel(urlPath);
```

**Note:** `loadModel()` calls this automatically when you pass a `require()` or `{ url }`, so you typically don't need to call this manually.

### `useLoadModel(source, options?)`

React hook for loading models with state management.

**Parameters:**
- `source`: Same as `loadModel()`
- `options`: `SessionOptions` (optional)

**Returns:** `OnnxRuntimePlugin`
```typescript
type OnnxRuntimePlugin = 
  | { model: InferenceSession; state: 'loaded' }
  | { model: undefined; state: 'loading' }
  | { model: undefined; state: 'error'; error: Error };
```

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

Manually free the session and release resources immediately.

**Note:** This is optional - sessions are automatically cleaned up by Nitro Modules when they go out of scope. Only call this if you need immediate memory cleanup.

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

## Exports

The library exports the following:

```typescript
import ort, { useLoadModel, copyFile } from 'react-native-nitro-onnxruntime';

// Using default export object
const session = await ort.loadModel(require('./model.onnx'));
const session2 = await ort.loadModelFromBuffer(arrayBuffer);

// Or using destructured methods
const { loadModel, loadModelFromBuffer } = ort;
const session3 = await loadModel(require('./model.onnx'));

// Utility functions
const modelPath = await copyFile(require('./model.onnx'));

// React hook
function MyComponent() {
  const modelState = useLoadModel(require('./model.onnx'));
  // ...
}
```

## Example App

See the [example](./example) directory for a complete working example with speed comparisons.

## Contributing

See the [contributing guide](CONTRIBUTING.md) to learn how to contribute to the repository and the development workflow.

## License

MIT

---

Made with [Nitro Modules](https://nitro.margelo.com/) and [create-react-native-library](https://github.com/callstack/react-native-builder-bob)
