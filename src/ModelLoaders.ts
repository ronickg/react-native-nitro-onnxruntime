import { NitroModules } from 'react-native-nitro-modules';
import type { ModelLoaderFactory } from './Onnxruntime.nitro';

/**
 * A factory for creating model loaders.
 */
export const ModelLoaders =
  NitroModules.createHybridObject<ModelLoaderFactory>('ModelLoaderFactory');
