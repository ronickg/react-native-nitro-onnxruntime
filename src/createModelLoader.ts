import { Image as RNImage } from 'react-native';
import type { ModelSource } from './ModelSource';
import type { InferenceSession, SessionOptions } from './Onnxruntime.nitro';
import { ModelLoaders } from './ModelLoaders';
import { OnnxRuntimes } from './OnnxRuntimes';

export function createModelLoader(
  source: ModelSource,
  options?: SessionOptions
): Promise<InferenceSession> {
  if (typeof source === 'number') {
    // It's a require(...) - a `number` which we need to resolve first
    const resolvedSource = RNImage.resolveAssetSource(source);
    if (resolvedSource.uri.startsWith('http')) {
      // In debug, assets are streamed over the network
      return createModelLoader({ url: resolvedSource.uri }, options);
    } else if (resolvedSource.uri.startsWith('file')) {
      // In release, assets are embedded files...
      return createModelLoader({ filePath: resolvedSource.uri }, options);
    } else {
      // ...or resource IDs
      return createModelLoader({ resource: resolvedSource.uri }, options);
    }
  } else if ('filePath' in source) {
    // It's a { filePath }
    return ModelLoaders.createFileModelLoader(source.filePath).then((path) =>
      OnnxRuntimes.loadModel(path, options)
    );
  } else if ('resource' in source) {
    // It's a { resource }
    return ModelLoaders.createResourceModelLoader(source.resource).then(
      (path) => OnnxRuntimes.loadModel(path, options)
    );
  } else if ('url' in source) {
    // It's a { url }
    return ModelLoaders.createUrlModelLoader(source.url).then((path) =>
      OnnxRuntimes.loadModel(path, options)
    );
  } else {
    throw new Error(`Unknown model source! ${JSON.stringify(source)}`);
  }
}
