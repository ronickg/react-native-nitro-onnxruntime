import type { HybridObject } from 'react-native-nitro-modules';
export interface EncodedTensor {
  readonly dims: readonly number[];
  readonly type: string;
  readonly data: ArrayBuffer;
}
export interface InferenceSession
  extends HybridObject<{ ios: 'c++'; android: 'c++' }> {
  readonly key: string;
  readonly inputNames: string[];
  readonly outputNames: string[];
  run(
    feeds: Record<string, EncodedTensor>
    // options: RunOptions
  ): Promise<Record<string, ArrayBuffer>>;
  close(): Promise<void>;
}

// Interface for ONNX Runtime in Nitro
export interface Onnxruntime
  extends HybridObject<{ ios: 'c++'; android: 'c++' }> {
  // Get the ONNX Runtime version
  getVersion(): string;

  // Create an inference session with a model file
  loadModel(
    modelPath: string
    // options?: SessionOptions
  ): Promise<InferenceSession>;

  loadModelFromBuffer(
    buffer: ArrayBuffer
    // options?: SessionOptions
  ): Promise<InferenceSession>;
}
