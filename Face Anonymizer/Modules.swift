//
//  modules.swift
//  Face Anonymizer
//
//  Created by Jose Bouza on 4/3/20.
//  Copyright © 2020 Jose Bouza. All rights reserved.
//

import UIKit
import CoreML

public struct PixelData {
    let a: UInt8 = 255
    var r: UInt8
    var g: UInt8
    var b: UInt8
    
}

func imageFromARGB32Bitmap(pixels: [PixelData], width: Int, height: Int) -> UIImage? {
    guard width > 0 && height > 0 else { return nil }
    guard pixels.count == width * height else { return nil }

    let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)
    let bitsPerComponent = 8
    let bitsPerPixel = 32

    var data = pixels // Copy to mutable []
    let data_ = NSData(bytes: &data, length: data.count*MemoryLayout<PixelData>.stride)
    guard let providerRef = CGDataProvider(data: data_)
        else { return nil }
    guard let cgim = CGImage(
        width: width,
        height: height,
        bitsPerComponent: bitsPerComponent,
        bitsPerPixel: bitsPerPixel,
        bytesPerRow: width * MemoryLayout<PixelData>.stride,
        space: rgbColorSpace,
        bitmapInfo: bitmapInfo,
        provider: providerRef,
        decode: nil,
        shouldInterpolate: true,
        intent: .defaultIntent
        )
        else { return nil }

    return UIImage(cgImage: cgim)
}

class DeepPrivacy {
    var deepprivacy_module : DeepPrivacyModule
    
    init(){
        deepprivacy_module = {
            if let filePath = Bundle.main.path(forResource: "deep_privacy", ofType: "pt"),
                    let module = DeepPrivacyModule(fileAtPath: filePath) {
                    return module
                } else {
                    fatalError("Can't find or process the model file!")
                }
            }()
        }
    
    func predict(image : UIImage, keypoints : MLMultiArray, bbox : MLMultiArray) -> UIImage{
        let resizedImage = image
        guard var pixelBuffer = resizedImage.normalized() else {
            fatalError("")
        }
        
        guard var outputs = deepprivacy_module.predict(image: UnsafeMutableRawPointer(&pixelBuffer), image_width: NSNumber(value: image.cgImage!.width), image_height: NSNumber(value: image.cgImage!.height), keypoints: keypoints.dataPointer, bbox: bbox.dataPointer, n: bbox.shape[0]) else {
            fatalError("")
        }
        
        let size = [NSNumber(value: image.cgImage!.height), NSNumber(value: image.cgImage!.width), NSNumber(3)]
        
        var outputs_array : [PixelData] = []
        for index in stride(from: 0, to: 3*image.cgImage!.height*image.cgImage!.width-2, by:3){
            outputs_array.append(PixelData(r: UInt8((outputs[index])), g:UInt8(outputs[index+1]), b:UInt8(outputs[index+2])))
        }
        
        return imageFromARGB32Bitmap(pixels: outputs_array, width: image.cgImage!.width, height: image.cgImage!.height)!
    }
    
}

class SSFDProcessing {
    var preprocess_module : SSFDPreProcessingModule
    var postprocess_module : SSFDPostProcessingModule
    init(){
        preprocess_module = {
            if let filePath = Bundle.main.path(forResource: "preprocess_ssfd", ofType: "pt"),
                let module = SSFDPreProcessingModule(fileAtPath: filePath) {
                return module
            } else {
                fatalError("Can't find or process the model file!")
            }
        }()
        
        postprocess_module = {
            if let filePath = Bundle.main.path(forResource: "postprocess_ssfd", ofType: "pt"),
                let module = SSFDPostProcessingModule(fileAtPath: filePath) {
                return module
            } else {
                fatalError("Can't find or process the model file!")
            }
        }()
    }
    
    func preprocess(image : UIImage) -> MLMultiArray{
        let resizedImage = image
        guard var pixelBuffer = resizedImage.normalized() else {
            fatalError("")
        }
                
        guard var outputs = preprocess_module.predict(image: UnsafeMutableRawPointer(&pixelBuffer), image_height:NSNumber(value: image.cgImage!.height), image_width:NSNumber(value:image.cgImage!.width)) else {
            fatalError("")
        }
        
        let size = [NSNumber(1), NSNumber(3), NSNumber(value: image.cgImage!.height), NSNumber(value: image.cgImage!.width)]
        
        let outputs_array = try? MLMultiArray(shape: size, dataType: MLMultiArrayDataType.float32)
        
        for index in 0...(3*image.cgImage!.height*image.cgImage!.width-1){
            outputs_array![index] = outputs[index]
        }
        
        return outputs_array!
    }
    
    func postprocess(olist : [MLMultiArray], ols1 : [NSNumber], ols2: [NSNumber]) -> MLMultiArray{
        // need to convert olist to std::vector<Tensor> in .mm code
        guard var outputs = postprocess_module.predict(ol1:olist[0].dataPointer,
                                                       ol2:olist[1].dataPointer,
                                                       ol3:olist[2].dataPointer,
                                                       ol4:olist[3].dataPointer,
                                                       ol5:olist[4].dataPointer,
                                                       ol6:olist[5].dataPointer,
                                                       ol7:olist[6].dataPointer,
                                                       ol8:olist[7].dataPointer,
                                                       ol9:olist[8].dataPointer,
                                                       ol10:olist[9].dataPointer,
                                                       ol11:olist[10].dataPointer,
                                                       ol12:olist[11].dataPointer,
                                                       ols1:ols1, ols2:ols2) else {
            fatalError("")
        }
        
        let numsamples = outputs.count/5
        
        let size = [NSNumber(value: numsamples), NSNumber(5)]
        let strides = [NSNumber(1), NSNumber(1)] // output is contiguous.
        
        let outputs_array = try? MLMultiArray(shape: size, dataType: MLMultiArrayDataType.float32)
        
        for index in 0...(5*numsamples-1){
            outputs_array![index] = outputs[index]
        }
        
        return outputs_array!
        
    }
}

class NetworkProcessing {
    var sfd_model : sfd_detector
    var sfd_processing_models : SSFDProcessing
    var keypoint_model : KeypointDataHandler
    var deepprivacy_model : DeepPrivacy
    
    init(){
        sfd_model = sfd_detector()
        sfd_processing_models = SSFDProcessing()
        keypoint_model = (try? KeypointDataHandler())!
        deepprivacy_model = DeepPrivacy()
    }
    
    private func fix_olist(olist_p : [MLMultiArray]) -> [MLMultiArray]{
        var out : [MLMultiArray] = []
        for i in 0...olist_p.count-1{
            let ol = olist_p[i]
            if ol.shape[0] == ol.shape[1]{
                out.append(try! ol.reshaped(to: [Int(ol.shape[1]), Int(ol.shape[2]), Int(ol.shape[3]), Int(ol.shape[4])]))
            }
            else{
                out.append(ol)
            }
        }
        
        return out
    }
    
    func predict(image : UIImage) -> UIImage{
        let image_ = image.resized(toWidth: 700)
        
        let start = DispatchTime.now() // <<<<<<<<<< Start time
        let sfd_pre = sfd_processing_models.preprocess(image: image_)
        
        let olist = try? sfd_model.prediction(input_img: sfd_pre)
        
        
        let olist_p_ = [olist!.ol1, olist!.ol2, olist!.ol3, olist!.ol4, olist!.ol5, olist!.ol6, olist!.ol7, olist!.ol8, olist!.ol9, olist!.ol10, olist!.ol11, olist!.ol12]
        let olist_p = fix_olist(olist_p: olist_p_)
        let olist_s2 = olist_p.map{$0.shape[$0.shape.count-1]}
        let olist_s1 = olist_p.map{$0.shape[$0.shape.count-2]}
        let sfd_out = sfd_processing_models.postprocess(olist: olist_p, ols1:olist_s1, ols2:olist_s2)
        let end = DispatchTime.now()
        let nanoTime = end.uptimeNanoseconds - start.uptimeNanoseconds // <<<<< Difference in nano seconds (UInt64)
        let timeInterval = Double(nanoTime) / 1_000_000_000
        print(timeInterval)
        
        print(sfd_out)
        let from_ = CGRect(x:0,y:0,width:image_.cgImage!.width,height:image_.cgImage!.height)
        let to_ = CGSize(width:image_.cgImage!.width, height:image_.cgImage!.height)
        // right now we use the code from the tf-examples github directly, so we convert the UIImage to
        // a CVPixelBuffer. It would be better to modify the code to take the UIImage.
        let start1 = DispatchTime.now() // <<<<<<<<<< Start time
        let keypoints = keypoint_model.runPoseNet(on: image_.pixelBuffer()!, from: from_, to: to_)
        let end1 = DispatchTime.now()
        let nanoTime1 = end1.uptimeNanoseconds - start1.uptimeNanoseconds // <<<<< Difference in nano seconds (UInt64)
        let timeInterval1 = Double(nanoTime1) / 1_000_000_000
        print(timeInterval1)
        
        let needed_keypoints = keypoints!.0.dots.prefix(7)
        print(needed_keypoints)
        let keypoints_out = try? MLMultiArray(shape: [NSNumber(1), NSNumber(7), NSNumber(2)], dataType: MLMultiArrayDataType.float32)

        for i in 0...6{
            keypoints_out![[0,NSNumber(value: i),0]] = NSNumber(value: Float(needed_keypoints[i].x))
            keypoints_out![[0,NSNumber(value: i),1]] = NSNumber(value: Float(needed_keypoints[i].y))
        }
        
        let start2 = DispatchTime.now() // <<<<<<<<<< Start time
        let image_out = deepprivacy_model.predict(image: image_, keypoints: keypoints_out!, bbox: sfd_out)
        let end2 = DispatchTime.now()
        let nanoTime2 = end2.uptimeNanoseconds - start2.uptimeNanoseconds // <<<<< Difference in nano seconds (UInt64)
        let timeInterval2 = Double(nanoTime2
            ) / 1_000_000_000
        print(timeInterval2)
        
        return image_out
    }
}
