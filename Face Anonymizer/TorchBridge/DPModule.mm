//
//  DPModule.m
//  Face Anonymizer
//
//  Created by Jose Bouza on 4/10/20.
//  Copyright © 2020 Jose Bouza. All rights reserved.
//
#import "DPModule.h"
#import <LibTorch/LibTorch.h>
#import <Foundation/Foundation.h>

#include <iostream>

@implementation DeepPrivacyModule {
 @protected
  torch::jit::script::Module _impl;
}

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath {
  self = [super init];
  if (self) {
    try {
      auto qengines = at::globalContext().supportedQEngines();
      if (std::find(qengines.begin(), qengines.end(), at::QEngine::QNNPACK) != qengines.end()) {
        at::globalContext().setQEngine(at::QEngine::QNNPACK);
      }
      _impl = torch::jit::load(filePath.UTF8String);
      _impl.eval();
    } catch (const std::exception& exception) {
      NSLog(@"%s", exception.what());
      return nil;
    }
  }
  return self;
}

- (nullable NSArray<NSNumber*>*)deepPrivacyGenerator:(void*)imageBuffer
                                           im_width : (NSNumber*)im_width
                                           im_height: (NSNumber*)im_height
                                           keypoints:(void*)keypoints
                                           bbox: (void*)bbox
                                           n: (NSNumber*)n {
  try {
      std::cout << "h " << [im_height intValue] << std::endl;
    at::Tensor tensor = torch::from_blob(imageBuffer, {3, [im_height intValue], [im_width intValue]}, at::kFloat);
      at::Tensor keypoints_ = torch::from_blob(keypoints, {[n intValue], 7, 2}, at::kFloat);
    at::Tensor bbox_ = torch::from_blob(bbox, {[n intValue], 4}, at::kFloat);
    at::Tensor n_ = torch::randn({1});
    n_[0] = [n intValue];
    at::Tensor z = torch::randn({1,32,4,4}, at::kFloat);
      
    torch::autograd::AutoGradMode guard(false);
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    auto outputTensor = _impl.forward({tensor, keypoints_, bbox_, n_, z}).toTensor();
    uint8_t* floatBuffer = outputTensor.data_ptr<uint8_t>();
    if (!floatBuffer) {
      return nil;
    }
    auto tensor_size = outputTensor.sizes();
          
      
    long long total_size = 1;
    for(auto size : tensor_size){
        total_size = total_size*size;
        std::cout << size << std::endl;
    }
    
    
    NSMutableArray* results = [[NSMutableArray alloc] init];
    for (int i = 0; i < total_size; i++) {
      [results addObject:@(floatBuffer[i])];
    }
    return [results copy];
  } catch (const std::exception& exception) {
    NSLog(@"%s", exception.what());
  }
  return nil;
}

@end
