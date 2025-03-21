import { NitroModules } from 'react-native-nitro-modules';
import type { NitroOnnxruntime } from './NitroOnnxruntime.nitro';

const NitroOnnxruntimeHybridObject =
  NitroModules.createHybridObject<NitroOnnxruntime>('NitroOnnxruntime');

export function multiply(a: number, b: number): number {
  return NitroOnnxruntimeHybridObject.multiply(a, b);
}
