import Foundation
import NitroModules

public class AssetManager: HybridAssetManagerSpec {

    public override init() {
        super.init()
    }

    public func copyFile(source: String) throws -> Promise<String> {
        return Promise.async {
            do {
                print("Copying file from source: \(source)...")

                // Get the document directory for storing files
                guard let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
                    throw NSError(domain: "AssetManager", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to get documents directory"])
                }

                // Extract the base filename without query parameters
                let baseFileName: String
                if source.contains("?") {
                    let components = source.components(separatedBy: "?")
                    baseFileName = URL(string: components[0])?.lastPathComponent ?? "unknown_file"
                } else {
                    baseFileName = URL(string: source)?.lastPathComponent ?? "unknown_file"
                }

                // Create destination file path
                let destinationFile = documentsDirectory.appendingPathComponent(baseFileName)

                // Create parent directory if needed
                let destinationDirectory = destinationFile.deletingLastPathComponent()
                try FileManager.default.createDirectory(at: destinationDirectory, withIntermediateDirectories: true)

                if !FileManager.default.fileExists(atPath: destinationFile.path) {
                    print("File doesn't exist, copying from source...")

                    if source.contains("://") {
                        let url = URL(string: source)!

                        switch url.scheme {
                        case "file":
                            // Copy from local file
                            try FileManager.default.copyItem(at: url, to: destinationFile)

                        case "http", "https":
                            // Download from network
                            let data = try Data(contentsOf: url)
                            try data.write(to: destinationFile)

                        default:
                            throw NSError(domain: "AssetManager", code: 3, userInfo: [NSLocalizedDescriptionKey: "Unsupported URI scheme: \(url.scheme ?? "unknown")"])
                        }
                    } else {
                        // For direct resource names, we'll look in the bundle
                        if let resourceURL = Bundle.main.url(forResource: source, withExtension: nil) {
                            try FileManager.default.copyItem(at: resourceURL, to: destinationFile)
                        } else {
                            throw NSError(domain: "AssetManager", code: 4, userInfo: [NSLocalizedDescriptionKey: "Resource not found in bundle: \(source)"])
                        }
                    }

                    print("File copied successfully to: \(destinationFile.path)")
                } else {
                    print("File already exists at destination: \(destinationFile.path)")
                }

                return destinationFile.path
            } catch {
                print("Error copying file: \(error.localizedDescription)")
                throw error
            }
        }
    }
}
