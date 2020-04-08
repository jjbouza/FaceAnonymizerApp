//
//  modules.swift
//  Face Anonymizer
//
//  Created by Jose Bouza on 4/3/20.
//  Copyright © 2020 Jose Bouza. All rights reserved.
//

import UIKit
import CoreML

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
        let strides = [NSNumber(1), NSNumber(1), NSNumber(1), NSNumber(1)] // output is contiguous.
        
        let outputs_array = try? MLMultiArray(shape: size, dataType: MLMultiArrayDataType.float32)
        
        for index in 0...(3*image.cgImage!.height*image.cgImage!.width-1){
            outputs_array![index] = outputs[index]
        }
        
        return outputs_array!
    }
    
    func postprocess(olist : [MLMultiArray], ols : [NSNumber]) -> MLMultiArray{
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
                                                       ols:ols) else {
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
    
    init(){
        sfd_model = sfd_detector()
        sfd_processing_models = SSFDProcessing()
        keypoint_model = (try? KeypointDataHandler())!
    }
    
    func predict(image : UIImage){
        let sfd_pre = sfd_processing_models.preprocess(image: image)
        let olist = try? sfd_model.prediction(input_img: sfd_pre)
        
        
        let olist_p = [olist!.ol1, olist!.ol2, olist!.ol3, olist!.ol4, olist!.ol5, olist!.ol6, olist!.ol7, olist!.ol8, olist!.ol9, olist!.ol10, olist!.ol11, olist!.ol12]
        
        let olist_s = olist_p.map{$0.shape[$0.shape.count-1]}
        
        let sfd_out = sfd_processing_models.postprocess(olist: olist_p, ols:olist_s)
        print(sfd_out)
        //let keypoints = keypointrcnn_model.predict(image: image)
        let from_ = CGRect(x:0,y:0,width:100,height:100)
        let to_ = CGSize(width:100, height:100)
        let keypoints = keypoint_model.runPoseNet(on: image.buffer()!, from: from_, to: to_)
        print(keypoints?.0)
    }
}
