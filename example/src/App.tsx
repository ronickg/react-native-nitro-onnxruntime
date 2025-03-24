import {
  Button,
  Platform,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  View,
} from 'react-native';
import ort from 'react-native-nitro-onnxruntime';
import RNFS from 'react-native-fs';
// @ts-ignore
import { InferenceSession as OnnxRuntimeInferenceSession } from 'onnxruntime-react-native';
// @ts-ignore
import { Tensor } from 'onnxruntime-common';
import { loadTensorflowModel } from 'react-native-fast-tflite';
import { useState, useEffect } from 'react';

const ModelPath = {
  YOLOV5: `${RNFS.DocumentDirectoryPath}/yolov5s.onnx`,
  YOLOV5_TFLITE: `${RNFS.DocumentDirectoryPath}/yolov5s-fp16.tflite`,
};

export default function App() {
  const [results, setResults] = useState<{ [key: string]: string }>({});
  const [modelOptions, setModelOptions] = useState({
    // iOS CoreML options
    useCoreML: false,
    useCPUOnly: false,
    useCPUAndGPU: false,
    enableOnSubgraph: false,
    onlyEnableDeviceWithANE: false,

    // Android NNAPI options
    useNNAPI: false,
    useFP16: false,
    useNCHW: false,
    cpuDisabled: false,
    cpuOnly: false,
  });

  const toggleOption = (option: keyof typeof modelOptions) => {
    setModelOptions((prev) => ({ ...prev, [option]: !prev[option] }));
  };

  const logResult = (key: string, value: string) => {
    setResults((prev) => ({ ...prev, [key]: value }));
  };

  // Check if models exist
  useEffect(() => {
    const checkModels = async () => {
      const modelsStatus = await Promise.all([
        RNFS.exists(ModelPath.YOLOV5),
        RNFS.exists(ModelPath.YOLOV5_TFLITE),
      ]);

      logResult(
        'Models Status',
        `YOLOv5 ONNX: ${modelsStatus[0] ? '✓' : '✗'}, YOLOv5 TFLite: ${modelsStatus[1] ? '✓' : '✗'}`
      );
    };

    checkModels();
  }, []);

  // Test 1: Load model from local file (require)
  const testLoadFromRequire = async () => {
    try {
      logResult('Load Model', 'Loading YOLOv5 from require...');
      const start = performance.now();

      // Note: In a real app, you would have the model in your assets
      await ort.loadModel(require('./yolov5s.onnx'));

      const end = performance.now();
      logResult(
        'Load from require',
        `Loaded in ${(end - start).toFixed(2)} ms`
      );
    } catch (error) {
      logResult('Load from require', `Error: ${(error as Error).message}`);
    }
  };

  // Test 2: Load model from URL (remote or file://)
  const testLoadFromURL = async () => {
    try {
      logResult('Load Model', 'Loading YOLOv5 from URL...');
      const start = performance.now();

      // You could use a remote URL or file:// URL
      await ort.loadModel({
        url: 'https://raw.githubusercontent.com/ronickg/react-native-nitro-onnxruntime/main/example/src/yolov5s.onnx',
      });

      const end = performance.now();
      logResult('Load from URL', `Loaded in ${(end - start).toFixed(2)} ms`);
    } catch (error) {
      logResult('Load from URL', `Error: ${(error as Error).message}`);
    }
  };

  // Test 3: Load model from direct file path
  const testLoadFromFilePath = async () => {
    try {
      logResult('Load Model', 'Loading YOLOv5 from file path...');
      const start = performance.now();

      // Direct path to the file without file:// prefix
      await ort.loadModel({
        url: 'file://' + ModelPath.YOLOV5,
      });

      const end = performance.now();
      logResult('Load from path', `Loaded in ${(end - start).toFixed(2)} ms`);
    } catch (error) {
      logResult('Load from path', `Error: ${(error as Error).message}`);
    }
  };

  // Performance test - Nitro ONNX
  const testNitroOnnxPerformance = async () => {
    try {
      logResult('Performance', 'Testing YOLOv5 with Nitro ONNX...');

      // if (!(await RNFS.exists(ModelPath.YOLOV5))) {
      //   throw new Error('YOLOv5 ONNX model not found');
      // }

      const options: any = {};

      // Add platform-specific options
      if (Platform.OS === 'ios' && modelOptions.useCoreML) {
        options.executionProviders = [
          {
            name: 'coreml',
            useCPUOnly: modelOptions.useCPUOnly,
            useCPUAndGPU: modelOptions.useCPUAndGPU,
            enableOnSubgraph: modelOptions.enableOnSubgraph,
            onlyEnableDeviceWithANE: modelOptions.onlyEnableDeviceWithANE,
          },
        ];
      } else if (Platform.OS === 'android' && modelOptions.useNNAPI) {
        options.executionProviders = [
          {
            name: 'nnapi',
            useFP16: modelOptions.useFP16,
            useNCHW: modelOptions.useNCHW,
            cpuDisabled: modelOptions.cpuDisabled,
            cpuOnly: modelOptions.cpuOnly,
          },
        ];
      }

      const model = await ort.loadModel(require('./yolov5s.onnx'), options);

      // Prepare input data - YOLOv5 takes [1, 3, 640, 640] input
      const inputData = new Float32Array(1 * 3 * 640 * 640).fill(0.1);

      // Warm-up run
      await model.run({ images: inputData.buffer });

      // Performance measurement
      const iterations = 5;
      let totalTime = 0;

      for (let i = 0; i < iterations; i++) {
        const start = performance.now();
        await model.run({ images: inputData.buffer });
        const end = performance.now();
        totalTime += end - start;
      }

      logResult('Nitro ONNX', `Avg: ${(totalTime / iterations).toFixed(2)} ms`);
    } catch (error) {
      logResult('Nitro ONNX', `Error: ${(error as Error).message}`);
    }
  };

  // Performance test - React Native ONNX
  const testReactNativeOnnxPerformance = async () => {
    try {
      logResult('Performance', 'Testing YOLOv5 with React Native ONNX...');

      // if (!(await RNFS.exists(ModelPath.YOLOV5))) {
      //   throw new Error('YOLOv5 ONNX model not found');
      // }

      const options: any = {};

      // Add platform-specific options
      if (Platform.OS === 'ios' && modelOptions.useCoreML) {
        options.executionProviders = [
          {
            name: 'coreml',
            useCPUOnly: modelOptions.useCPUOnly,
            useCPUAndGPU: modelOptions.useCPUAndGPU,
            enableOnSubgraph: modelOptions.enableOnSubgraph,
            onlyEnableDeviceWithANE: modelOptions.onlyEnableDeviceWithANE,
          },
        ];
      } else if (Platform.OS === 'android' && modelOptions.useNNAPI) {
        options.executionProviders = [
          {
            name: 'nnapi',
            useFP16: modelOptions.useFP16,
            useNCHW: modelOptions.useNCHW,
            cpuDisabled: modelOptions.cpuDisabled,
            cpuOnly: modelOptions.cpuOnly,
          },
        ];
      }

      const buffer = await ort.loadBufferFromSource(require('./yolov5s.onnx'));

      const session = await OnnxRuntimeInferenceSession.create(buffer, options);

      // Prepare input data - YOLOv5 takes [1, 3, 640, 640] input
      const inputData = new Float32Array(1 * 3 * 640 * 640).fill(0.1);
      const tensorInput = new Tensor('float32', inputData, [1, 3, 640, 640]);

      // Warm-up run
      await session.run({ images: tensorInput });

      // Performance measurement
      const iterations = 5;
      let totalTime = 0;

      for (let i = 0; i < iterations; i++) {
        const start = performance.now();
        await session.run({ images: tensorInput });
        const end = performance.now();
        totalTime += end - start;
      }

      logResult('RN ONNX', `Avg: ${(totalTime / iterations).toFixed(2)} ms`);
      await session.release();
    } catch (error) {
      logResult('RN ONNX', `Error: ${(error as Error).message}`);
    }
  };

  // Performance test - TFLite
  const testTFLitePerformance = async () => {
    try {
      logResult('Performance', 'Testing YOLOv5 with TFLite...');

      // if (!(await RNFS.exists(ModelPath.YOLOV5_TFLITE))) {
      //   throw new Error('YOLOv5 TFLite model not found');
      // }

      // Note: In a real app, this would be a valid asset
      const model = await loadTensorflowModel(
        require('./yolov5s-fp16.tflite'),
        Platform.OS === 'android'
          ? modelOptions.useNNAPI
            ? 'nnapi'
            : 'default'
          : modelOptions.useCoreML
            ? 'core-ml'
            : 'default'
      );

      // Prepare input data (TFLite typically uses NHWC format)
      const inputData = new Float32Array(1 * 640 * 640 * 3).fill(0.1);

      // Warm-up run
      await model.run([inputData]);

      // Performance measurement
      const iterations = 5;
      let totalTime = 0;

      for (let i = 0; i < iterations; i++) {
        const start = performance.now();
        await model.run([inputData]);
        const end = performance.now();
        totalTime += end - start;
      }

      logResult('TFLite', `Avg: ${(totalTime / iterations).toFixed(2)} ms`);
    } catch (error) {
      logResult('TFLite', `Error: ${(error as Error).message}`);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>YOLOv5 Model Tests</Text>
      <ScrollView style={styles.scrollView}>
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Model Loading Tests</Text>
          <View style={{ flexDirection: 'row', gap: 16 }}>
            <Button title="require()" onPress={testLoadFromRequire} />
            <Button title="url" onPress={testLoadFromURL} />
            <Button title="file://" onPress={testLoadFromFilePath} />
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Performance Tests</Text>
          <View style={{ flexDirection: 'row', gap: 16 }}>
            <Button title="Nitro ONNX" onPress={testNitroOnnxPerformance} />
            <Button
              title="React Native ONNX"
              onPress={testReactNativeOnnxPerformance}
            />
            <Button title="TFLite" onPress={testTFLitePerformance} />
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Model Options</Text>

          {Platform.OS === 'ios' && (
            <>
              <View style={styles.optionRow}>
                <Text>Use CoreML</Text>
                <Switch
                  value={modelOptions.useCoreML}
                  onValueChange={() => toggleOption('useCoreML')}
                />
              </View>

              {modelOptions.useCoreML && (
                <>
                  <View style={styles.optionRow}>
                    <Text>CPU Only</Text>
                    <Switch
                      value={modelOptions.useCPUOnly}
                      onValueChange={() => toggleOption('useCPUOnly')}
                    />
                  </View>

                  <View style={styles.optionRow}>
                    <Text>CPU And GPU</Text>
                    <Switch
                      value={modelOptions.useCPUAndGPU}
                      onValueChange={() => toggleOption('useCPUAndGPU')}
                    />
                  </View>

                  <View style={styles.optionRow}>
                    <Text>Enable On Subgraph</Text>
                    <Switch
                      value={modelOptions.enableOnSubgraph}
                      onValueChange={() => toggleOption('enableOnSubgraph')}
                    />
                  </View>

                  <View style={styles.optionRow}>
                    <Text>Only On Devices with ANE</Text>
                    <Switch
                      value={modelOptions.onlyEnableDeviceWithANE}
                      onValueChange={() =>
                        toggleOption('onlyEnableDeviceWithANE')
                      }
                    />
                  </View>
                </>
              )}
            </>
          )}

          {Platform.OS === 'android' && (
            <>
              <View style={styles.optionRow}>
                <Text>Use NNAPI</Text>
                <Switch
                  value={modelOptions.useNNAPI}
                  onValueChange={() => toggleOption('useNNAPI')}
                />
              </View>

              {modelOptions.useNNAPI && (
                <>
                  <View style={styles.optionRow}>
                    <Text>Use FP16</Text>
                    <Switch
                      value={modelOptions.useFP16}
                      onValueChange={() => toggleOption('useFP16')}
                    />
                  </View>

                  <View style={styles.optionRow}>
                    <Text>Use NCHW</Text>
                    <Switch
                      value={modelOptions.useNCHW}
                      onValueChange={() => toggleOption('useNCHW')}
                    />
                  </View>

                  <View style={styles.optionRow}>
                    <Text>CPU Disabled</Text>
                    <Switch
                      value={modelOptions.cpuDisabled}
                      onValueChange={() => toggleOption('cpuDisabled')}
                    />
                  </View>

                  <View style={styles.optionRow}>
                    <Text>CPU Only</Text>
                    <Switch
                      value={modelOptions.cpuOnly}
                      onValueChange={() => toggleOption('cpuOnly')}
                    />
                  </View>
                </>
              )}
            </>
          )}
        </View>

        <View style={styles.resultsContainer}>
          <Text style={styles.sectionTitle}>Results</Text>
          {Object.entries(results).map(([key, value]) => (
            <View key={key} style={styles.resultRow}>
              <Text style={styles.resultKey}>{key}:</Text>
              <Text style={styles.resultValue}>{value}</Text>
            </View>
          ))}
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: 'white',
    flex: 1,
    padding: 16,
  },
  scrollView: {
    flex: 1,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  section: {
    marginBottom: 24,
    gap: 8,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  optionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
  },
  resultsContainer: {
    backgroundColor: '#f5f5f5',
    padding: 16,
    borderRadius: 8,
    marginBottom: 24,
  },
  resultRow: {
    flexDirection: 'row',
    marginVertical: 4,
  },
  resultKey: {
    fontWeight: 'bold',
    marginRight: 8,
  },
  resultValue: {
    flex: 1,
  },
});
