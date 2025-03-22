import {
  Button,
  // Image,
  PermissionsAndroid,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import {
  ort,
  type EncodedTensor,
  // type InferenceSession,
} from 'react-native-nitro-onnxruntime';
// import ortD from 'react-native-nitro-onnxruntime';
import RNFS from 'react-native-fs';
// @ts-ignore
import { InferenceSession as OnnxRuntimeInferenceSession } from 'onnxruntime-react-native';
import { Tensor } from 'onnxruntime-common';

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
      // Load the model using the ArrayBuffer
      // const source = require('./model.onnx');
      // const asset = Image.resolveAssetSource(source);
      // let uri = asset.uri;
      // console.log(`Resolved Model path: ${asset.uri}`);

      const files = await RNFS.readDir(RNFS.DocumentDirectoryPath);
      const file = files.find((_file) => _file.name === 'model.onnx');
      if (file && file.isFile()) {
        const start = performance.now();
        const model = await ort.loadModel(
          RNFS.DocumentDirectoryPath + '/' + file.name
        );
        const end = performance.now();
        console.log(`Model loaded in ${end - start} milliseconds`);
        // prepare inputs. a tensor need its corresponding TypedArray as data
        const dataA = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        const dataB = new Float32Array([
          10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
        ]);
        const tensorA: EncodedTensor = {
          type: 'float32',
          data: dataA.buffer,
          dims: [3, 4],
        };
        const tensorB: EncodedTensor = {
          type: 'float32',
          data: dataB.buffer,
          dims: [4, 3],
        };

        // prepare feeds. use model input names as keys.
        const feeds = { a: tensorA, b: tensorB };

        const result = await model.run(feeds);
        console.log('Result:', result);
      } else {
        console.log('Model not found');
      }
      // console.log('Model loaded successfully:', model);
    } catch (error) {
      console.error('Error loading model:', error);
    }
  };
  return (
    <View style={styles.container}>
      <Text style={styles.title}>ONNX Runtime Test</Text>
      <Text style={styles.versionText}>Version: {ort.getVersion()}</Text>
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
          const model = await ort.loadModel(
            RNFS.DocumentDirectoryPath + '/model.onnx'
          );
          // prepare inputs. a tensor need its corresponding TypedArray as data
          const dataA = new Float32Array([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
          ]);
          const dataB = new Float32Array([
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
          ]);
          const tensorA: EncodedTensor = {
            type: 'float32',
            data: dataA.buffer,
            dims: [3, 4],
          };
          const tensorB: EncodedTensor = {
            type: 'float32',
            data: dataB.buffer,
            dims: [4, 3],
          };

          // prepare feeds. use model input names as keys.
          const feeds = { a: tensorA, b: tensorB };

          const start = performance.now();
          const result = await model.run(feeds);
          const end = performance.now();
          console.log(`Model ran in ${end - start} milliseconds`);
          console.log('Result:', result.c);
          await model.close();
        }}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
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
