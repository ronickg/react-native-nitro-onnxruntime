// class NitroOnnxruntime: HybridNitroOnnxruntimeSpec {
//     public func multiply(a: Double, b: Double) throws -> Double {
//         return a * b
//     }
// }

import Foundation
import NitroModules

public class AssetManager: HybridAssetManagerSpec {

    public override init() {
        super.init()
    }

    public func fetchByteDataFromUrl(url: String) throws -> Promise<ArrayBufferHolder> {
        return Promise.async {
            do {
                print("Loading byte data from URL: \(url)...")

                var nsURL: URL?

                if url.contains("://") {
                    // It's a URL
                    guard let parsedURL = URL(string: url) else {
                        throw NSError(domain: "AssetManager", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid URL: \(url)"])
                    }
                    nsURL = parsedURL
                    print("Parsed URL: \(nsURL!)")
                } else {
                    // It's a resource name in the bundle
                    guard let resourceURL = Bundle.main.url(forResource: url, withExtension: nil) else {
                        throw NSError(domain: "AssetManager", code: 2, userInfo: [NSLocalizedDescriptionKey: "Resource not found: \(url)"])
                    }
                    nsURL = resourceURL
                    print("Resource URL: \(nsURL!)")
                }

                // Load the data
                let data: Data

                if nsURL!.isFileURL {
                    print("It's a file URL")
                    data = try Data(contentsOf: nsURL!, options: .mappedIfSafe)
                } else {
                    print("It's a network URL/http resource")
                    data = try Data(contentsOf: nsURL!)
                }

                // Create ArrayBuffer from Data
                let buffer = try ArrayBufferHolder.copy(data: data)

                return buffer
            } catch {
                print("Error fetching byte data: \(error.localizedDescription)")
                throw error
            }
        }
    }
}
