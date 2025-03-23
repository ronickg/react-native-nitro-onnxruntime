import type { HybridObject } from 'react-native-nitro-modules';
// import type {
//   OnnxModelOptions,
//   InferenceSession as OnnxRuntimeInferenceSession,
//   OnnxValueDataLocation,
// } from 'onnxruntime-common';

export interface Tensor {
  readonly dims: readonly number[];
  readonly type: string;
  readonly name: string;
}

// Define a named interface for the object structure
export interface ExternalDataEntry {
  data: string | ArrayBuffer;
  path: string;
}

// Update ExternalDataFileType to use the named interface
export type ExternalDataFileType = ExternalDataEntry | string;

export interface SessionOptions {
  // Execution providers
  executionProviders?: string[]; // e.g., ['cpu', 'nnapi']

  // Optimization level (0-99, default is 99)
  optimizationLevel?: number;

  // Enable/disable memory pattern optimization
  enableMemoryPattern?: boolean;

  // Number of threads to use
  intraOpNumThreads?: number;
  interOpNumThreads?: number;

  // Graph optimization level: 0=disable, 1=basic, 2=extended, 3=all
  graphOptimizationLevel?: number;

  // External data loading - for models that store weights externally
  externalDataPaths?: string[];

  // Logging level: 0=verbose, 1=info, 2=warning, 3=error, 4=fatal
  logSeverityLevel?: number;

  // Execution mode: 0=sequential, 1=parallel
  executionMode?: number;
}

export interface InferenceSession
  extends HybridObject<{ ios: 'c++'; android: 'c++' }> {
  readonly inputNames: Tensor[];
  readonly outputNames: Tensor[];
  run(
    feeds: Record<string, ArrayBuffer>
    // options: RunOptions
  ): Promise<Record<string, ArrayBuffer>>;
}

// Interface for ONNX Runtime in Nitro
export interface Onnxruntime
  extends HybridObject<{ ios: 'c++'; android: 'c++' }> {
  // Get the ONNX Runtime version
  getVersion(): string;

  // Create an inference session with a model file
  loadModel(
    modelPath: string,
    options?: SessionOptions
  ): Promise<InferenceSession>;

  loadModelFromBuffer(
    buffer: ArrayBuffer,
    options?: SessionOptions
  ): Promise<InferenceSession>;
}
