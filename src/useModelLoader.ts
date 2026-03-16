import { useEffect, useState } from 'react';
import type { ModelSource } from './ModelSource';
import { createModelLoader } from './createModelLoader';
import type { InferenceSession, SessionOptions } from './Onnxruntime.nitro';

type Result =
  | { state: 'loading'; model: undefined; error: undefined }
  | { state: 'ready'; model: InferenceSession; error: undefined }
  | { state: 'error'; model: undefined; error: Error };

/**
 * A hook to asynchronously load an ONNX model from the
 * given {@linkcode ModelSource}.
 * @example
 * ```ts
 * const { model, error } = useModel(require('./model.onnx'))
 * ```
 */
export function useModelLoader(
  source: ModelSource,
  options?: SessionOptions
): Result {
  const [result, setResult] = useState<Result>({
    state: 'loading',
    model: undefined,
    error: undefined,
  });

  useEffect(() => {
    const load = async () => {
      try {
        const model = await createModelLoader(source, options);
        setResult({ state: 'ready', model, error: undefined });
      } catch (e) {
        const error = e instanceof Error ? e : new Error(`${e}`);
        setResult({ state: 'error', model: undefined, error });
      }
    };
    load();
  }, [source, options]);

  return result;
}
