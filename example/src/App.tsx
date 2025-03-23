import {
  Button,
  // Image,
  PermissionsAndroid,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import ort from 'react-native-nitro-onnxruntime';

// import ortD from 'react-native-nitro-onnxruntime';
import RNFS from 'react-native-fs';
// @ts-ignore
import { InferenceSession as OnnxRuntimeInferenceSession } from 'onnxruntime-react-native';
import { Tensor } from 'onnxruntime-common';
import { loadTensorflowModel } from 'react-native-fast-tflite';

export default function App() {
  const requestStoragePermission = async () => {
    try {
      const granted = await PermissionsAndroid.requestMultiple([
        PermissionsAndroid.PERMISSIONS.READ_EXTERNAL_STORAGE,
        PermissionsAndroid.PERMISSIONS.WRITE_EXTERNAL_STORAGE,
      ]);

      if (
        granted['android.permission.READ_EXTERNAL_STORAGE'] ===
          PermissionsAndroid.RESULTS.GRANTED &&
        granted['android.permission.WRITE_EXTERNAL_STORAGE'] ===
          PermissionsAndroid.RESULTS.GRANTED
      ) {
        console.log('Storage permissions granted');
        await loadModel();
      } else {
        console.log('Storage permissions denied');
      }
    } catch (err) {
      console.warn(err);
    }
  };

  const loadModel = async () => {
    try {
      // testAssetManager(require('./model.onnx'));
    } catch (error) {
      console.error('Error loading model:', error);
    }
  };
  return (
    <View style={styles.container}>
      <Text style={styles.title}>ONNX Runtime Test</Text>
      <Button
        title="Request Storage Permission"
        onPress={requestStoragePermission}
      />
      <Button title="Load Model" onPress={loadModel} />
      <Button
        title="Load Big Model"
        onPress={async () => {
          const start = performance.now();
          await ort.loadModel(
            RNFS.DocumentDirectoryPath + '/arcfaceresnet100-8.onnx'
          );
          const end = performance.now();
          console.log(`Model loaded in ${end - start} milliseconds`);
        }}
      />
      <Button
        title="Old Speed Test"
        onPress={async () => {
          // Create inference session using the backend
          const session = await OnnxRuntimeInferenceSession.create(
            RNFS.DocumentDirectoryPath + '/model.onnx'
            // { backendHint: 'android' } // Pass the custom backend
          );

          // Prepare input data
          const dataA = new Float32Array([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
          ]);
          const dataB = new Float32Array([
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
          ]);

          // Create Tensor objects directly (no need for manual JSIBlob conversion)
          const tensorA = new Tensor('float32', dataA, [3, 4]);
          const tensorB = new Tensor('float32', dataB, [4, 3]);

          // Prepare feeds using model input names
          const feeds = {
            a: tensorA,
            b: tensorB,
          };

          const start = performance.now();

          // Run the model
          const results = await session.run(feeds);

          const end = performance.now();
          console.log(`Model ran in ${end - start} milliseconds`);

          // Access results (already decoded as Tensor objects)
          // Example: results['outputName'].data will give you the output data
          console.log('Results:', results);

          // Clean up
          await session.release();
        }}
      />
      <Button
        title="New Speed Test"
        onPress={async () => {
          // const model = await ort.loadModel(
          //   RNFS.DocumentDirectoryPath + '/model.onnx'
          // );
          // // prepare inputs. a tensor need its corresponding TypedArray as data
          // const dataA = new Float32Array([
          //   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
          // ]);
          // const dataB = new Float32Array([
          //   10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
          // ]);
          // const tensorA: EncodedTensor = {
          //   type: 'float32',
          //   data: dataA.buffer,
          //   dims: [3, 4],
          // };
          // const tensorB: EncodedTensor = {
          //   type: 'float32',
          //   data: dataB.buffer,
          //   dims: [4, 3],
          // };
          // // prepare feeds. use model input names as keys.
          // const feeds = { a: tensorA, b: tensorB };
          // const start = performance.now();
          // const result = await model.run(feeds);
          // const end = performance.now();
          // console.log(`Model ran in ${end - start} milliseconds`);
          // console.log('Result:', result.c);
          // await model.close();
        }}
      />
      <Button
        title="Test SqueezeNet Model"
        onPress={async () => {
          console.log('Testing SqueezeNet Model');
          try {
            const modelPath = `${RNFS.DocumentDirectoryPath}/squeezenet1.1-7.onnx`;

            // Check if model exists
            const fileExists = await RNFS.exists(modelPath);
            if (!fileExists) {
              throw new Error(
                'SqueezeNet model not found. Please copy squeezenet1.1-7.onnx to the document directory.'
              );
            }

            // Create inference session
            const session = await OnnxRuntimeInferenceSession.create(
              modelPath,
              {
                // backendHint: onnxruntimeBackend, // Uncomment if using custom backend
              }
            );

            // Log input names for confirmation (optional)
            console.log('Model input names:', session.inputNames); // Should log ["data"]

            // Prepare dummy input (random data for [1, 3, 224, 224])
            const inputData = new Float32Array(1 * 3 * 224 * 224).map(() =>
              Math.random()
            );
            const tensorInput = new Tensor(
              'float32',
              inputData,
              [1, 3, 224, 224]
            );

            // Use the correct input name 'data'
            const feeds = {
              data: tensorInput,
            };

            // Warm-up run
            await session.run(feeds);

            // Measure performance
            const iterations = 10;
            let totalTime = 0;

            for (let i = 0; i < iterations; i++) {
              const start = performance.now();
              const results = await session.run(feeds);
              const end = performance.now();
              totalTime += end - start;

              if (i === 0) {
                console.log(
                  'Sample output shape:',
                  results[session.outputNames[0]].dims
                );
              }
            }

            const avgTime = totalTime / iterations;
            console.log(
              `Average runtime over ${iterations} iterations: ${avgTime.toFixed(2)} ms`
            );

            // Clean up
            await session.release();
          } catch (error) {
            console.error('Error running model:', error);
          }
        }}
      />
      {/* <Button
        title="Test SqueezeNet Model"
        onPress={async () => {
          console.log('Testing SqueezeNet Model');
          try {
            const modelPath = `${RNFS.DocumentDirectoryPath}/squeezenet1.1-7.onnx`;

            // Check if model exists
            const fileExists = await RNFS.exists(modelPath);
            if (!fileExists) {
              throw new Error(
                'SqueezeNet model not found. Please copy squeezenet1.1-7.onnx to the document directory.'
              );
            }

            // Create inference session
            const session = await ort.loadModel(modelPath);

            // Log input names for confirmation (optional)
            console.log('Model input names:', session.inputNames); // Should log ["data"]

            // Prepare dummy input (random data for [1, 3, 224, 224])
            const inputData = new Float32Array(1 * 3 * 224 * 224).map(() =>
              Math.random()
            );
            // const tensorInput = new Tensor(
            //   'float32',
            //   inputData,
            //   [1, 3, 224, 224]
            // );

            const tensorInput: EncodedTensor = {
              type: 'float32',
              data: inputData.buffer,
              dims: [1, 3, 224, 224],
            };

            // Use the correct input name 'data'
            const feeds = {
              data: tensorInput,
            };

            // Warm-up run
            await session.run(feeds);

            // Measure performance
            const iterations = 10;
            let totalTime = 0;

            for (let i = 0; i < iterations; i++) {
              const start = performance.now();
              const results = await session.run(feeds);
              const end = performance.now();
              totalTime += end - start;

              // if (i === 0) {
              //   console.log(
              //     'Sample output shape:',
              //     results[session.outputNames[0]].dims
              //   );
              // }
            }

            const avgTime = totalTime / iterations;
            console.log(
              `Average runtime over ${iterations} iterations: ${avgTime.toFixed(2)} ms`
            );

            // Clean up
            await session.close();
          } catch (error) {
            console.error('Error running model:', error);
          }
        }}
      /> */}
      <Button
        title="Test old YOLOv5 ONNX"
        onPress={async () => {
          console.log('Testing old YOLOv5 ONNX');
          try {
            const modelPath = `${RNFS.DocumentDirectoryPath}/yolov5s.onnx`;
            if (!(await RNFS.exists(modelPath))) {
              throw new Error('YOLOv5 ONNX model not found.');
            }
            const session = await OnnxRuntimeInferenceSession.create(
              modelPath,
              {
                executionProviders: [
                  {
                    name: 'nnapi',
                    useFP16: true,
                    useNCHW: false,
                    cpuDisabled: false,
                  },
                  // { name: 'xnnpack' },
                  { name: 'cpu' },
                ],
              }
            );
            console.log(session);

            // Prepare input [1, 3, 640, 640] (channel-first)
            const inputData = new Float32Array(1 * 3 * 640 * 640).map(() =>
              Math.random()
            );
            const tensorInput = new Tensor(
              'float32',
              inputData,
              [1, 3, 640, 640]
            );

            const feeds = { images: tensorInput }; // YOLOv5 ONNX input is 'images'

            // Warm-up
            await session.run(feeds);

            // Measure performance
            const iterations = 10;
            let totalTime = 0;
            for (let i = 0; i < iterations; i++) {
              const start = performance.now();
              await session.run(feeds);
              const end = performance.now();
              totalTime += end - start;
              if (i === 0) {
                // console.log(results[session.outputNames[0]]);
                // console.log(
                //   'ONNX output shape:',
                //   results[session.outputNames[0]].dims
                // );
              }
            }

            console.log(
              `ONNX avg runtime: ${(totalTime / iterations).toFixed(2)} ms`
            );
            await session.release();
          } catch (error) {
            console.error('ONNX error:', error);
          }
        }}
      />
      <Button
        title="Test new YOLOv5 ONNX"
        onPress={async () => {
          console.log('Testing new YOLOv5 ONNX');
          console.log(RNFS.DocumentDirectoryPath);
          try {
            const modelPath = `${RNFS.DocumentDirectoryPath}/yolov5s.onnx`;
            if (!(await RNFS.exists(modelPath))) {
              throw new Error('YOLOv5 ONNX model not found.');
            }

            // const options: InferenceSession.SessionOptions = {
            //   executionProviders: [
            //     // {
            //     //   name: 'nnapi',
            //     //   useFP16: true,
            //     //   useNCHW: false,
            //     //   // cpuDisabled: false,
            //     // },
            //   ],
            //   // logSeverityLevel: 0,
            // };
            console.log(modelPath);
            // const session = await ort.loadModel1(
            //   { url: 'file://' + modelPath },
            //   {
            //     executionProviders: [{ name: 'nnapi' }],
            //   }
            // );
            // const session = await ort.loadModel1(require('./yolov5s.onnx'), {
            //   executionProviders: [{ name: 'nnapi' }],
            // });
            const session = await ort.loadModel1(
              {
                url: 'https://1drv.ms/u/c/48a1c40521b4fd64/EWOdAias5mRMrCtT2Uo0wOAB230933CukBR1jZs2PHLy2g?e=iXGRdy',
              },
              {
                executionProviders: [{ name: 'nnapi' }],
              }
            );
            console.log(session.inputNames);
            console.log(session.outputNames);
            // Prepare input [1, 3, 640, 640] (channel-first)
            const inputData = new Float32Array(1 * 3 * 640 * 640).map(() =>
              Math.random()
            );
            // const tensorInput: EncodedTensor = {
            //   type: 'float32',
            //   data: inputData.buffer,
            //   dims: [1, 3, 640, 640],
            // };

            const feeds = { images: inputData.buffer }; // YOLOv5 ONNX input is 'images'

            // Warm-up
            await session.run(feeds);

            // Measure performance
            const iterations = 10;
            let totalTime = 0;
            for (let i = 0; i < iterations; i++) {
              const start = performance.now();
              await session.run(feeds);
              const end = performance.now();
              totalTime += end - start;
              if (i === 0) {
                // console.log(
                //   new Float32Array(results[session.outputNames[0]!.name]!)
                //     .length
                // );
              }
            }

            console.log(
              `ONNX avg runtime: ${(totalTime / iterations).toFixed(2)} ms`
            );
            // session.dispose();
            // await session.close();
          } catch (error) {
            console.error('ONNX error:', error);
          }
        }}
      />
      <Button
        title="Test YOLOv5 Tflite"
        onPress={async () => {
          // console.log('Testing YOLOv5 TFLite with react-native-fast-tflite');
          try {
            const modelPath = `${RNFS.DocumentDirectoryPath}/yolov5s-fp16.tflite`;
            if (!(await RNFS.exists(modelPath))) {
              throw new Error('YOLOv5 TFLite model not found.');
            }
            // console.log('Model path:', modelPath);
            // Load the model
            const model = await loadTensorflowModel(
              require('./yolov5s-fp16.tflite')
            );

            // console.log(model);
            // Prepare input [1, 640, 640, 3]
            const inputData = new Float32Array(1 * 640 * 640 * 3).map(() =>
              Math.random()
            );
            const inputs = [inputData]; // Wrap in array for run method

            // Warm-up run
            await model.run(inputs);

            // Measure performance
            const iterations = 10;
            let totalTime = 0;
            for (let i = 0; i < iterations; i++) {
              const start = performance.now();
              await model.run(inputs);
              const end = performance.now();
              totalTime += end - start;
              if (i === 0) {
                // const firstOutput = outputs;
                // console.log('TFLite output length:', firstOutput[0]?.length); // First output tensor length
              }
            }

            console.log(
              `TFLite avg runtime: ${(totalTime / iterations).toFixed(2)} ms`
            );
            // No explicit cleanup needed per docs, model is garbage-collected
          } catch (error) {
            console.error('TFLite error:', error);
          }
        }}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    gap: 10,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  versionText: {
    fontSize: 16,
    marginBottom: 20,
  },
  buttonContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  resultContainer: {
    padding: 10,
    backgroundColor: '#f0f0f0',
    borderRadius: 5,
    marginBottom: 20,
  },
  resultTitle: {
    fontWeight: 'bold',
    marginBottom: 5,
  },
  logContainer: {
    flex: 1,
    backgroundColor: '#f9f9f9',
    padding: 10,
    borderRadius: 5,
    marginTop: 10,
  },
  logTitle: {
    fontWeight: 'bold',
    marginBottom: 5,
  },
  logEntry: {
    fontSize: 12,
    marginBottom: 3,
  },
});
