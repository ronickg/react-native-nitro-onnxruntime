import { NitroModules } from 'react-native-nitro-modules';
import type { Onnxruntime } from './Onnxruntime.nitro';

/**
 * The ONNX Runtime native module.
 */
export const OnnxRuntimes =
  NitroModules.createHybridObject<Onnxruntime>('Onnxruntime');
