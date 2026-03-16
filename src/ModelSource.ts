import type { InferenceSession } from './Onnxruntime.nitro';

export type RequireType = number;

export type ModelSource =
  | { filePath: string }
  | { url: string }
  | { resource: string }
  | RequireType;

export function isInferenceSession(obj: unknown): obj is InferenceSession {
  // @ts-expect-error checking for hybrid object methods
  return typeof obj === 'object' && obj != null && obj.run != null;
}
