import { NitroModules } from 'react-native-nitro-modules';
import type { Onnxruntime, AssetManager } from './Onnxruntime.nitro';
import type { InferenceSession } from 'onnxruntime-common';
import { Image } from 'react-native';
import { useEffect, useState } from 'react';

const ort = NitroModules.createHybridObject<Onnxruntime>('Onnxruntime');
const assetManager =
  NitroModules.createHybridObject<AssetManager>('AssetManager');

type SessionOptions = Omit<
  InferenceSession.SessionOptions,
  | 'freeDimensionOverrides'
  | 'optimizedModelFilePath'
  | 'enableProfiling'
  | 'profileFilePrefix'
  | 'logVerbosityLevel'
  | 'preferredOutputLocation'
  | 'enableGraphCapture'
  | 'extra'
  | 'externalData'
>;

type Require = number; // ReturnType<typeof require>
type ModelSource = Require | { url: string };

export type OnnxRuntimePlugin =
  | {
      model: InferenceSession;
      state: 'loaded';
    }
  | {
      model: undefined;
      state: 'loading';
    }
  | {
      model: undefined;
      error: Error;
      state: 'error';
    };

async function loadBufferFromSource(source: ModelSource) {
  let uri: string;
  if (typeof source === 'number') {
    const asset = Image.resolveAssetSource(source);
    uri = asset.uri;
  } else if (typeof source === 'object' && 'url' in source) {
    uri = source.url;
  } else {
    throw new Error(
      'Onnx-runtime: Invalid source passed! Source should be either a React Native require(..) or a `{ url: string }` object!'
    );
  }
  return await assetManager.fetchByteDataFromUrl(uri);
}

async function loadModel(source: ModelSource, options?: SessionOptions) {
  const buffer = await loadBufferFromSource(source);
  //@ts-ignore Allowing the use of the SessionOptions type which is fully compatible with the nitro types
  return ort.loadModelFromBuffer(buffer, options);
}

export function useLoadModel(source: ModelSource, options?: SessionOptions) {
  const [state, setState] = useState<OnnxRuntimePlugin>({
    model: undefined,
    state: 'loading',
  });

  useEffect(() => {
    const load = async (): Promise<void> => {
      try {
        setState({ model: undefined, state: 'loading' });
        const m = await loadModel(source, options);
        //@ts-ignore Allowing the use of the SessionOptions type which is fully compatible with the nitro types
        setState({ model: m, state: 'loaded' });
      } catch (e) {
        console.error(`Failed to load Onnx-runtime Model ${source}!`, e);
        setState({ model: undefined, state: 'error', error: e as Error });
      }
    };
    load();
  }, [source, options]);

  return state;
}

export default {
  useLoadModel,
  loadModel,
  loadBufferFromSource,
};
