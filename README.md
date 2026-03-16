# react-native-nitro-onnxruntime

High-performance ONNX Runtime bindings for React Native, built with [Nitro Modules](https://nitro.margelo.com/) for maximum performance.

## Features

- **Blazing Fast**: Built with Nitro Modules for zero-overhead JSI bindings
- **Hardware Acceleration**: Support for NNAPI (Android), CoreML (iOS), and XNNPACK
- **Modern API**: Promise-based async API with TypeScript support
- **Flexible Model Loading**: Load models from files, URLs, resources, or `require()`
- **Full Type Support**: Complete TypeScript definitions
- **Configurable**: Extensive session options for optimization

## Installation

```sh
npm install react-native-nitro-onnxruntime react-native-nitro-modules
```

> **Note**: `react-native-nitro-modules` is required as this library relies on [Nitro Modules](https://nitro.margelo.com/).

## Usage

### Basic Example

```typescript
import { createModelLoader } from 'react-native-nitro-onnxruntime';

// Load a model
const session = await createModelLoader(require('./assets/model.onnx'));

// Get input/output information
console.log('Inputs:', session.inputNames);
console.log('Outputs:', session.outputNames);

// Prepare input data
const inputData = new Float32Array(1 * 3 * 224 * 224);
// ... fill inputData with your data

// Run inference
const results = session.run({
  [session.inputNames[0].name]: inputData.buffer,
});

// Access output
const outputBuffer = results[session.outputNames[0].name];
const outputData = new Float32Array(outputBuffer);
console.log('Output:', outputData);
```

### Model Sources

Models can be loaded from various sources using `createModelLoader()`:

```typescript
import { createModelLoader } from 'react-native-nitro-onnxruntime';

// From bundled asset (require)
const session = await createModelLoader(require('./assets/model.onnx'));

// From file path
const session = await createModelLoader({ filePath: '/path/to/model.onnx' });

// From URL (downloaded and cached)
const session = await createModelLoader({ url: 'https://example.com/model.onnx' });

// From bundle resource
const session = await createModelLoader({ resource: 'model.onnx' });
```

When using `require()`, the library automatically resolves the asset for both debug (metro HTTP) and release (embedded file/resource) builds — following the same pattern as [react-native-nitro-image](https://github.com/margelo/react-native-nitro-image).

All file-based sources are copied to the app's documents directory and cached. Subsequent loads use the cached file.

### Hardware Acceleration

#### Android (NNAPI)

```typescript
const session = await createModelLoader(require('./model.onnx'), {
  executionProviders: ['nnapi'],
});

// Or with options
const session = await createModelLoader(require('./model.onnx'), {
  executionProviders: [
    {
      name: 'nnapi',
      useFP16: true,
      cpuDisabled: true,
    },
  ],
});
```

#### iOS (CoreML)

```typescript
const session = await createModelLoader(require('./model.onnx'), {
  executionProviders: ['coreml'],
});

// Or with options
const session = await createModelLoader(require('./model.onnx'), {
  executionProviders: [
    {
      name: 'coreml',
      useCPUOnly: false,
      onlyEnableDeviceWithANE: true,
    },
  ],
});
```

#### XNNPACK (Cross-platform)

```typescript
const session = await createModelLoader(require('./model.onnx'), {
  executionProviders: ['xnnpack'],
});
```

### Advanced Configuration

```typescript
const session = await createModelLoader(require('./model.onnx'), {
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
  executionProviders: ['nnapi', 'cpu'],
});
```

### React Hook

The library provides a convenient React hook for loading models:

```typescript
import { useModelLoader } from 'react-native-nitro-onnxruntime';

function MyComponent() {
  const { state, model, error } = useModelLoader(require('./assets/model.onnx'), {
    executionProviders: ['nnapi'],
  });

  if (state === 'loading') {
    return <Text>Loading model...</Text>;
  }

  if (state === 'error') {
    return <Text>Error: {error.message}</Text>;
  }

  // state === 'ready'
  const runInference = async () => {
    const input = new Float32Array(224 * 224 * 3);
    const results = model.run({
      [model.inputNames[0].name]: input.buffer,
    });
  };

  return <Button onPress={runInference} title="Run Inference" />;
}
```

### Loading from Buffer

For advanced use cases (encrypted storage, authenticated endpoints, decompression), you can load models directly from an ArrayBuffer using the `OnnxRuntimes` object:

```typescript
import { OnnxRuntimes } from 'react-native-nitro-onnxruntime';

const response = await fetch('https://example.com/model.onnx');
const arrayBuffer = await response.arrayBuffer();

const session = await OnnxRuntimes.loadModelFromBuffer(arrayBuffer, {
  executionProviders: ['coreml'],
});
```

### Memory Management

Sessions are automatically cleaned up by Nitro Modules when they go out of scope. You can manually dispose of a session to free memory immediately:

```typescript
session.dispose();
```

You don't need to call `dispose()` in most cases — Nitro Modules will automatically clean up when the session is no longer referenced.

## API Reference

### `createModelLoader(source, options?)`

Load an ONNX model from various sources.

**Parameters:**

- `source`: `ModelSource` — The model source:
  - `number`: `require()` asset (auto-resolved for debug/release)
  - `{ filePath: string }`: Absolute file path on disk
  - `{ url: string }`: URL to download from (cached)
  - `{ resource: string }`: Platform bundle resource name
- `options`: `SessionOptions` (optional) — Configuration options

**Returns:** `Promise<InferenceSession>`

### `useModelLoader(source, options?)`

React hook for loading models with state management.

**Parameters:**

- `source`: `ModelSource`
- `options`: `SessionOptions` (optional)

**Returns:**

```typescript
type Result =
  | { state: 'loading'; model: undefined; error: undefined }
  | { state: 'ready'; model: InferenceSession; error: undefined }
  | { state: 'error'; model: undefined; error: Error };
```

### `OnnxRuntimes`

The underlying ONNX Runtime native module, exposed for advanced use cases.

```typescript
OnnxRuntimes.getVersion(): string;
OnnxRuntimes.loadModel(path: string, options?: SessionOptions): Promise<InferenceSession>;
OnnxRuntimes.loadModelFromBuffer(buffer: ArrayBuffer, options?: SessionOptions): Promise<InferenceSession>;
```

### `InferenceSession`

#### `session.inputNames` / `session.outputNames`

Arrays of tensor metadata:

```typescript
type Tensor = {
  name: string;
  dims: number[]; // Shape, negative values indicate dynamic dimensions
  type: string; // 'float32', 'int64', etc.
};
```

#### `session.run(feeds)`

Run synchronous inference.

- `feeds`: `Record<string, ArrayBuffer>` — Map of input names to ArrayBuffers
- **Returns:** `Record<string, ArrayBuffer>`

#### `session.runAsync(feeds)`

Run async inference.

- `feeds`: `Record<string, ArrayBuffer>`
- **Returns:** `Promise<Record<string, ArrayBuffer>>`

#### `session.dispose()`

Manually free the session and release resources immediately.

### `ModelSource`

```typescript
type ModelSource =
  | number // require('model.onnx')
  | { filePath: string } // absolute path
  | { url: string } // download URL
  | { resource: string }; // bundle resource
```

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

- Android (API 21+)
- iOS (13.0+)

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

## Example App

See the [example](./example) directory for a complete working example with speed comparisons.

## Contributing

See the [contributing guide](CONTRIBUTING.md) to learn how to contribute to the repository and the development workflow.

## License

MIT

---

Made with [Nitro Modules](https://nitro.margelo.com/) and [create-react-native-library](https://github.com/callstack/react-native-builder-bob)
