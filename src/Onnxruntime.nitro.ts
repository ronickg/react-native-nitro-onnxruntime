import type { HybridObject } from 'react-native-nitro-modules';
export interface Tensor {
  readonly dims: readonly number[];
  readonly type: string;
  readonly name: string;
}
interface ProviderOptions {
  name: string; // Discriminant union of literal strings
  useCPUOnly?: boolean; // CoreML-specific
  useCPUAndGPU?: boolean; // CoreML-specific
  enableOnSubgraph?: boolean; // CoreML-specific
  onlyEnableDeviceWithANE?: boolean; // CoreML-specific
  useFP16?: boolean; // NNAPI-specific
  useNCHW?: boolean; // NNAPI-specific
  cpuDisabled?: boolean; // NNAPI-specific
  cpuOnly?: boolean; // NNAPI-specific
}

type ExecutionProvider = string | ProviderOptions;
interface SessionOptions {
  intraOpNumThreads?: number;
  interOpNumThreads?: number;
  graphOptimizationLevel?: string;
  enableCpuMemArena?: boolean;
  enableMemPattern?: boolean;
  executionMode?: string;
  executionProviders?: ExecutionProvider[];
  logId?: string;
  logSeverityLevel?: number;
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
  getVersion(): string;

  loadModel(
    modelPath: string,
    options?: SessionOptions
  ): Promise<InferenceSession>;

  loadModelFromBuffer(
    buffer: ArrayBuffer,
    options?: SessionOptions
  ): Promise<InferenceSession>;
}

export interface AssetManager
  extends HybridObject<{ ios: 'swift'; android: 'kotlin' }> {
  copyFile(source: string): Promise<string>;
}
