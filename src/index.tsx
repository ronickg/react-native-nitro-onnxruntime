import { NitroModules } from 'react-native-nitro-modules';
import type { Onnxruntime, AssetManager } from './Onnxruntime.nitro';
import type { InferenceSession } from 'onnxruntime-common';
import { Image } from 'react-native';

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

function loadModel(modelPath: string, options?: SessionOptions) {
  //@ts-ignore Allowing the use of the SessionOptions type which is fully compatible with the nitro types
  return ort.loadModel(modelPath, options);
}

type Require = number; // ReturnType<typeof require>
type ModelSource = Require | { url: string };

// export function testAssetManager(source: ModelSource) {
//   let uri: string;
//   if (typeof source === 'number') {
//     console.log(`Loading Tensorflow Lite Model ${source}`);
//     const asset = Image.resolveAssetSource(source);
//     uri = asset.uri;
//     console.log(`Resolved Model path: ${asset.uri}`);
//   } else if (typeof source === 'object' && 'url' in source) {
//     uri = source.url;
//   } else {
//     throw new Error(
//       'TFLite: Invalid source passed! Source should be either a React Native require(..) or a `{ url: string }` object!'
//     );
//   }
//   const buffer = assetManager.fetchByteDataFromUrl(uri);
//   console.log(new Uint8Array(buffer).length);
// }

function loadModel1(source: ModelSource, options?: SessionOptions) {
  let uri: string;
  console.log(source);
  if (typeof source === 'number') {
    console.log(`Loading Tensorflow Lite Model ${source}`);
    const asset = Image.resolveAssetSource(source);
    uri = asset.uri;
    console.log(`Resolved Model path: ${asset.uri}`);
  } else if (typeof source === 'object' && 'url' in source) {
    uri = source.url;
  } else {
    throw new Error(
      'TFLite: Invalid source passed! Source should be either a React Native require(..) or a `{ url: string }` object!'
    );
  }
  const buffer = assetManager.fetchByteDataFromUrl(uri);
  console.log(new Uint8Array(buffer).length);
  //@ts-ignore Allowing the use of the SessionOptions type which is fully compatible with the nitro types
  return ort.loadModelFromBuffer(buffer, options);
}

export default {
  loadModel,
  loadModel1,
  // testAssetManager,
};
