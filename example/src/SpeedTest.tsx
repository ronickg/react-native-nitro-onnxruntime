import { Button, Platform, ScrollView, Text, View } from 'react-native';
import ort from 'react-native-nitro-onnxruntime';
// import {
//   loadTensorflowModel,
//   type TensorflowModel,
// } from 'react-native-fast-tflite';
import { useState } from 'react';

// @ts-ignore
import { InferenceSession } from 'onnxruntime-react-native';
import { Tensor } from 'onnxruntime-common';

interface ModelConfig {
  url: string;
  shape?: number[];
}

interface TestResult {
  loadTime: number;
  avgTime: number;
  minTime: number;
  maxTime: number;
}

const OnnxModels: Record<string, ModelConfig> = {
  yolov5s: {
    url: require('./models/yolov5s.onnx'),
  },
  mobilenetv2: {
    url: require('./models/mobilenetv2.onnx'),
  },
  resnet18: {
    url: require('./models/resnet18.onnx'),
  },
  mobilenetv3: {
    url: require('./models/mobilenetv3_large.onnx'),
  },
};

const OnnxOldModels: Record<string, ModelConfig> = {
  yolov5s: {
    url: 'file:///data/data/nitroonnxruntime.example/files/yolov5s.onnx',
    shape: [1, 3, 640, 640], // CHW - unchanged, works as is
  },
  mobilenetv2: {
    url: 'file:///data/data/nitroonnxruntime.example/files/mobilenetv2.onnx',
    shape: [1, 224, 224, 3], // HWC - corrected
  },
  resnet18: {
    url: 'file:///data/data/nitroonnxruntime.example/files/resnet18.onnx',
    shape: [1, 224, 224, 3], // HWC - corrected
  },
  mobilenetv3: {
    url: 'file:///data/data/nitroonnxruntime.example/files/mobilenetv3_large.onnx',
    shape: [1, 224, 224, 3], // HWC - corrected
  },
};

// const TFLiteModels: Record<string, any> = {
//   // yolov5s: {
//   //   url: 'file:///data/data/nitroonnxruntime.example/files/yolov5s-fp16.tflite',
//   // },
//   // mobilenetv2: {
//   //   url: 'file:///data/data/nitroonnxruntime.example/files/mobilenetv2.tflite',
//   // },
//   // resnet18: {
//   //   url: 'file:///data/data/nitroonnxruntime.example/files/resnet18.tflite',
//   // },
//   // mobilenetv3: {
//   //   url: 'file:///data/data/nitroonnxruntime.example/files/mobilenetv3_large.tflite',
//   // },
//   yolov5s: require('./models/tfyolov5s-fp16.tflite'),
//   mobilenetv2: require('./models/tfmobilenetv2.tflite'),
//   resnet18: require('./models/tfresnet18.tflite'),
//   mobilenetv3: require('./models/tfmobilenetv3_large.tflite'),
// };

const runOnnxModelTest = async (modelName: string, config: ModelConfig) => {
  console.log(`Testing ${modelName} (ONNX)`);
  try {
    const loadStart = performance.now();
    const session = await ort.loadModel(config.url);
    const loadTime = performance.now() - loadStart;

    console.log(`Input names: ${JSON.stringify(session.inputNames)}`);
    if (!session.inputNames || session.inputNames.length === 0) {
      throw new Error('No input names found in model');
    }

    // Use the found tensor or default to the first one
    const tensor = session.inputNames[0];
    if (!tensor) {
      throw new Error('No valid input tensor found in model');
    }

    console.log(`Using input tensor: ${JSON.stringify(tensor)}`);

    // Handle potentially negative dimensions by replacing with positive ones
    const cleanDims = tensor.dims.map((dim) => (dim < 0 ? 1 : dim));
    console.log(`Adjusted dimensions: ${JSON.stringify(cleanDims)}`);

    // Calculate total elements
    const totalElements = cleanDims.reduce((a: number, b: number) => a * b, 1);
    console.log(`Creating Float32Array with ${totalElements} elements`);

    const inputData = new Float32Array(totalElements).fill(0.1);

    console.log('Warm-up run (ONNX)...');
    await session.run({ [tensor.name]: inputData.buffer });

    const iterations = 5;
    let totalTime = 0;
    const timings: number[] = [];

    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      await session.run({ [tensor.name]: inputData.buffer });
      const end = performance.now();
      const iterationTime = end - start;
      timings.push(iterationTime);
      totalTime += iterationTime;
    }

    const avgTime = totalTime / iterations;
    const minTime = Math.min(...timings);
    const maxTime = Math.max(...timings);

    return { loadTime, avgTime, minTime, maxTime };
  } catch (error) {
    console.error(`Error testing ${modelName} (ONNX):`, error);
    throw error;
  }
};

const runOnnxOldModelTest = async (modelName: string, config: ModelConfig) => {
  console.log(`Testing ${modelName} (ONNX)`);
  try {
    const loadStart = performance.now();
    // const session = await ort.loadModel(config.url);
    const session = await InferenceSession.create(config.url);
    const loadTime = performance.now() - loadStart;

    console.log(`Input names: ${JSON.stringify(session.handler.inputNames)}`);
    if (
      !session.handler.inputNames ||
      session.handler.inputNames.length === 0
    ) {
      throw new Error('No input names found in model');
    }

    const inputName = session.handler.inputNames[0];

    // Use the found tensor or default to the first one
    // const tensor = session.inputNames[0];
    // if (!tensor) {
    //   throw new Error('No valid input tensor found in model');
    // }

    // console.log(`Using input tensor: ${JSON.stringify(tensor)}`);

    // Handle potentially negative dimensions by replacing with positive ones
    const cleanDims = config.shape?.map((dim) => (dim < 0 ? 1 : dim)) ?? [];
    console.log(`Adjusted dimensions: ${JSON.stringify(cleanDims)}`);

    // Calculate total elements
    const totalElements = cleanDims.reduce((a: number, b: number) => a * b, 1);
    console.log(`Creating Float32Array with ${totalElements} elements`);

    const inputData = new Float32Array(totalElements).fill(0.1);
    const tensorInput = new Tensor('float32', inputData, config.shape);

    console.log('Warm-up run (ONNX)...');
    await session.run({ [inputName]: tensorInput });

    const iterations = 5;
    let totalTime = 0;
    const timings: number[] = [];

    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      await session.run({ [inputName]: tensorInput });
      const end = performance.now();
      const iterationTime = end - start;
      timings.push(iterationTime);
      totalTime += iterationTime;
    }

    const avgTime = totalTime / iterations;
    const minTime = Math.min(...timings);
    const maxTime = Math.max(...timings);

    return { loadTime, avgTime, minTime, maxTime };
  } catch (error) {
    console.error(`Error testing ${modelName} (ONNX):`, error);
    throw error;
  }
};

// const runTfliteModelTest = async (modelName: string) => {
//   console.log(`Testing ${modelName} (TFLite)`);
//   try {
//     const loadStart = performance.now();
//     const model: TensorflowModel = await loadTensorflowModel(
//       TFLiteModels[modelName] as any
//     );
//     const loadTime = performance.now() - loadStart;

//     console.log(`Input shape: ${JSON.stringify(model.inputs[0]?.shape)}`);

//     if (!model.inputs || model.inputs.length === 0) {
//       throw new Error('No inputs found in TFLite model');
//     }

//     // Get the first input shape and ensure no negative dimensions
//     const inputShape = model.inputs[0]?.shape || [];
//     const cleanShape = inputShape.map((dim: number) => (dim < 0 ? 1 : dim));
//     console.log(`Adjusted dimensions: ${JSON.stringify(cleanShape)}`);

//     const totalElements =
//       cleanShape.length > 0
//         ? cleanShape.reduce((a: number, b: number) => a * b, 1)
//         : 1;

//     console.log(`Creating Float32Array with ${totalElements} elements`);
//     const inputData = new Float32Array(totalElements).fill(0.1);

//     console.log('Warm-up run (TFLite)...');
//     await model.run([inputData]);

//     const iterations = 5;
//     let totalTime = 0;
//     const timings: number[] = [];

//     for (let i = 0; i < iterations; i++) {
//       const start = performance.now();
//       await model.run([inputData]);
//       const end = performance.now();
//       const iterationTime = end - start;
//       timings.push(iterationTime);
//       totalTime += iterationTime;
//     }

//     const avgTime = totalTime / iterations;
//     const minTime = Math.min(...timings);
//     const maxTime = Math.max(...timings);

//     return { loadTime, avgTime, minTime, maxTime };
//   } catch (error) {
//     console.error(`Error testing ${modelName} (TFLite):`, error);
//     throw error;
//   }
// };

export default function SpeedTestScreen() {
  const [results, setResults] = useState<
    Record<
      string,
      { tflite?: TestResult; onnx?: TestResult; onnxOld?: TestResult }
    >
  >({});

  const handleOnnxTest = async (modelName: string) => {
    try {
      const result = await runOnnxModelTest(modelName, OnnxModels[modelName]!);
      setResults((prev) => ({
        ...prev,
        [modelName]: {
          ...prev[modelName],
          onnx: result,
        },
      }));
    } catch (error) {
      // Error logged in runOnnxModelTest
    }
  };

  const handleOldOnnxTest = async (modelName: string) => {
    try {
      const result = await runOnnxOldModelTest(
        modelName,
        OnnxOldModels[modelName]!
      );
      setResults((prev) => ({
        ...prev,
        [modelName]: {
          ...prev[modelName],
          onnxOld: result,
        },
      }));
    } catch (error) {
      // Error logged in runOnnxModelTest
    }
  };

  // const handleTfliteTest = async (modelName: string) => {
  //   try {
  //     const result = await runTfliteModelTest(modelName);
  //     setResults((prev) => ({
  //       ...prev,
  //       [modelName]: {
  //         ...prev[modelName],
  //         tflite: result,
  //       },
  //     }));
  //   } catch (error) {
  //     // Error logged in runTfliteModelTest
  //   }
  // };

  return (
    <View
      style={{
        flex: 1,
        paddingTop: Platform.OS === 'ios' ? 36 : 0,
      }}
    >
      <View
        style={{
          flexDirection: 'row',
          gap: 10,
          alignItems: 'center',
          marginBottom: 4,
        }}
      >
        {/* <Text style={{ fontSize: 18, fontWeight: 'bold' }}>TFLite:</Text> */}
        <Text style={{ fontSize: 18, fontWeight: 'bold' }}>ONNX (old):</Text>

        <View style={{ flex: 1 }} />
        <Text style={{ fontSize: 18, fontWeight: 'bold' }}>ONNX (new):</Text>
      </View>
      <ScrollView style={{ flex: 1, width: '100%' }}>
        {Object.keys(OnnxModels).map((model) => {
          const tflite = results[model]?.tflite;
          const onnx = results[model]?.onnx;
          const onnxOld = results[model]?.onnxOld;
          return (
            <View key={model} style={{ gap: 6 }}>
              <View
                style={{ flexDirection: 'row', gap: 10, alignItems: 'center' }}
              >
                <Text style={{ fontSize: 18, fontWeight: 'bold' }}>
                  {model}
                </Text>
                <View style={{ flex: 1 }} />
                {/* <Button
                  title="Test TFLite"
                  onPress={() => handleTfliteTest(model)}
                /> */}
                <Button
                  title="Test ONNX (old)"
                  onPress={() => handleOldOnnxTest(model)}
                />
                <Button
                  title="Test ONNX (new)"
                  onPress={() => handleOnnxTest(model)}
                />
              </View>
              <View
                style={{
                  flexDirection: 'row',
                  justifyContent: 'space-between',
                  padding: 2,
                }}
              >
                <View style={{ flex: 1 }}>
                  {tflite && (
                    <>
                      <Text>Load: {tflite.loadTime.toFixed(2)}ms</Text>
                      <Text>Avg: {tflite.avgTime.toFixed(2)}ms</Text>
                      <Text>Min: {tflite.minTime.toFixed(2)}ms</Text>
                      <Text>Max: {tflite.maxTime.toFixed(2)}ms</Text>
                    </>
                  )}
                  {onnxOld && (
                    <>
                      <Text>Load: {onnxOld.loadTime.toFixed(2)}ms</Text>
                      <Text>Avg: {onnxOld.avgTime.toFixed(2)}ms</Text>
                      <Text>Min: {onnxOld.minTime.toFixed(2)}ms</Text>
                      <Text>Max: {onnxOld.maxTime.toFixed(2)}ms</Text>
                    </>
                  )}
                </View>
                <View style={{ flex: 1, alignItems: 'flex-end' }}>
                  {onnx && (
                    <>
                      <Text>Load: {onnx.loadTime.toFixed(2)}ms</Text>
                      <Text>Avg: {onnx.avgTime.toFixed(2)}ms</Text>
                      <Text>Min: {onnx.minTime.toFixed(2)}ms</Text>
                      <Text>Max: {onnx.maxTime.toFixed(2)}ms</Text>
                    </>
                  )}
                </View>
              </View>
              {tflite && onnx && (
                <View style={{ padding: 2 }}>
                  <Text>Differences (TFLite - ONNX):</Text>
                  <Text
                    style={{
                      color:
                        tflite.loadTime - onnx.loadTime < 0 ? 'red' : 'green',
                    }}
                  >
                    Load: {(tflite.loadTime - onnx.loadTime).toFixed(2)}ms
                  </Text>
                  <Text
                    style={{
                      color:
                        tflite.avgTime - onnx.avgTime < 0 ? 'red' : 'green',
                    }}
                  >
                    Avg: {(tflite.avgTime - onnx.avgTime).toFixed(2)}ms
                  </Text>
                  <Text
                    style={{
                      color:
                        tflite.minTime - onnx.minTime < 0 ? 'red' : 'green',
                    }}
                  >
                    Min: {(tflite.minTime - onnx.minTime).toFixed(2)}ms
                  </Text>
                  <Text
                    style={{
                      color:
                        tflite.maxTime - onnx.maxTime < 0 ? 'red' : 'green',
                    }}
                  >
                    Max: {(tflite.maxTime - onnx.maxTime).toFixed(2)}ms
                  </Text>
                </View>
              )}
              {onnx && onnxOld && (
                <View style={{ padding: 2 }}>
                  <Text>Differences (ONNX new - ONNX old):</Text>
                  <Text
                    style={{
                      color:
                        onnx.loadTime - onnxOld.loadTime < 0 ? 'green' : 'red',
                    }}
                  >
                    Load: {(onnx.loadTime - onnxOld.loadTime).toFixed(2)}ms
                  </Text>
                  <Text
                    style={{
                      color:
                        onnx.avgTime - onnxOld.avgTime < 0 ? 'green' : 'red',
                    }}
                  >
                    Avg: {(onnx.avgTime - onnxOld.avgTime).toFixed(2)}ms
                  </Text>
                  <Text
                    style={{
                      color:
                        onnx.minTime - onnxOld.minTime < 0 ? 'green' : 'red',
                    }}
                  >
                    Min: {(onnx.minTime - onnxOld.minTime).toFixed(2)}ms
                  </Text>
                  <Text
                    style={{
                      color:
                        onnx.maxTime - onnxOld.maxTime < 0 ? 'green' : 'red',
                    }}
                  >
                    Max: {(onnx.maxTime - onnxOld.maxTime).toFixed(2)}ms
                  </Text>
                </View>
              )}
            </View>
          );
        })}
      </ScrollView>
    </View>
  );
}
