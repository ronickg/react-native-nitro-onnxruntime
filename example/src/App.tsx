import { StyleSheet, View } from 'react-native';
// @ts-ignore
// @ts-ignore
import SpeedTest from './SpeedTest';

export default function App() {
  return (
    <View style={styles.container}>
      <SpeedTest />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: 'white',
    flex: 1,
    padding: 8,
  },
});
