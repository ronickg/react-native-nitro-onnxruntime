import type { HybridObject } from 'react-native-nitro-modules';
// import type { InferenceSession as OnnxRuntimeInferenceSession } from 'onnxruntime-common';
export interface Tensor {
  readonly dims: readonly number[];
  readonly type: string;
  readonly name: string;
}
export interface InferenceSession
  extends HybridObject<{ ios: 'c++'; android: 'c++' }> {
  readonly key: string;
  readonly inputNames: Tensor[];
  readonly outputNames: Tensor[];
  run(
    feeds: Record<string, ArrayBuffer>
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
    // options?: OnnxRuntimeInferenceSession.SessionOptions
  ): Promise<InferenceSession>;
}
