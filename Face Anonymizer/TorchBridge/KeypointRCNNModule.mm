//
//  KeypointRCNNModule.m
//  Face Anonymizer
//
//  Created by Jose Bouza on 4/3/20.
//  Copyright © 2020 Jose Bouza. All rights reserved.
//

#import "KeypointRCNNModule.h"
#import <LibTorch/LibTorch.h>

@implementation KeypointRCNNModule {
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

- (NSArray<NSNumber*>*)predict_keypointrcnn:(void*)imageBuffer {
  try {
    at::Tensor tensor = torch::from_blob(imageBuffer, {3, 450, 450}, at::kFloat);
    torch::autograd::AutoGradMode guard(false);
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    auto outputTensor = _impl.forward({tensor}).toTensor();
    float* floatBuffer = outputTensor.data_ptr<float>();
    if (!floatBuffer) {
      return nil;
    }
      
    auto tensor_size = outputTensor.sizes();
      
    long long total_size = 1;
    for(auto size : tensor_size)
        total_size = total_size*size;
      
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
