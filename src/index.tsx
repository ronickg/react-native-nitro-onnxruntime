import { NitroModules } from 'react-native-nitro-modules';
import type { Onnxruntime } from './Onnxruntime.nitro';
export * from './Onnxruntime.nitro';

// const OnnxruntimeHybridObject =
//   NitroModules.createHybridObject<Onnxruntime>('Onnxruntime');

// export function getVersion(): string {
//   return OnnxruntimeHybridObject.getVersion();
// }

export const ort = NitroModules.createHybridObject<Onnxruntime>('Onnxruntime');
