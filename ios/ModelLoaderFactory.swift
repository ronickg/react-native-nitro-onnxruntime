import Foundation
import NitroModules

class ModelLoaderFactory: HybridModelLoaderFactorySpec {

    // MARK: - Private Helpers

    private func modelsDirectory() -> URL {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            .appendingPathComponent("onnx_models")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func extractFileName(from source: String) -> String {
        let clean = source.contains("?") ? String(source.prefix(upTo: source.firstIndex(of: "?")!)) : source
        return URL(string: clean)?.lastPathComponent ?? "unknown_file"
    }

    // MARK: - HybridModelLoaderFactorySpec

    func createFileModelLoader(filePath rawFilePath: String) throws -> Promise<String> {
        return Promise.async {
            // Strip file:// prefix if present
            let filePath = rawFilePath.replacingOccurrences(of: "file://", with: "")
            guard FileManager.default.fileExists(atPath: filePath) else {
                throw RuntimeError.error(withMessage: "Model file not found at path: \(filePath)")
            }
            return filePath
        }
    }

    func createResourceModelLoader(name: String) throws -> Promise<String> {
        return Promise.async {
            guard let resourceURL = Bundle.main.url(forResource: name, withExtension: nil) else {
                throw RuntimeError.error(withMessage: "Model resource not found in bundle: \(name)")
            }
            return resourceURL.path
        }
    }

    func createUrlModelLoader(url urlString: String) throws -> Promise<String> {
        return Promise.async {
            let fileName = self.extractFileName(from: urlString)
            let destination = self.modelsDirectory().appendingPathComponent(fileName)

            // Return cached file if it exists
            guard !FileManager.default.fileExists(atPath: destination.path) else {
                return destination.path
            }

            guard let url = URL(string: urlString) else {
                throw RuntimeError.error(withMessage: "Invalid URL: \(urlString)")
            }

            let data = try Data(contentsOf: url)
            try data.write(to: destination)
            return destination.path
        }
    }
}
