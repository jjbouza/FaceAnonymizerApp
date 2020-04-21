#import "SSFDModule.h"
#import <LibTorch/LibTorch.h>
#import <Foundation/Foundation.h>

#include <iostream>

@implementation SSFDPreProcessingModule {
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

- (NSArray<NSNumber*>*)preprocessImage_ssfd:(void*)imageBuffer
                                  im_height:(NSNumber*)im_height
                                  im_width: (NSNumber*)im_width {
  try {
    at::Tensor tensor = torch::from_blob(imageBuffer, {1, 3, [im_height intValue], [im_width intValue]}, at::kFloat);
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


@implementation SSFDPostProcessingModule {
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

- (NSArray<NSNumber*>*)postprocessImage_ssfd:(void*)ol1
                                                    ol2:(void*)ol2
                                                    ol3:(void*)ol3
                                                    ol4: (void*)ol4
                                                    ol5: (void*)ol5
                                                    ol6: (void*)ol6
                                                    ol7:(void*)ol7
                                                    ol8:(void*)ol8
                                                    ol9: (void*)ol9
                                                    ol10: (void*)ol10
                                                    ol11: (void*)ol11
                                                    ol12: (void*)ol12
                                                    ols1: (NSArray<NSNumber*>*)ols1
                                                    ols2: (NSArray<NSNumber*>*)ols2

{
  try {
    at::Tensor tensor1 = torch::from_blob(ol1,   {1, 2, [ols1[0] intValue], [ols2[0] intValue]}, at::kFloat);
    at::Tensor tensor2 = torch::from_blob(ol2,   {1, 4, [ols1[1] intValue], [ols2[1] intValue]}, at::kFloat);
    at::Tensor tensor3 = torch::from_blob(ol3,   {1, 2, [ols1[2] intValue], [ols2[2] intValue]}, at::kFloat);
    at::Tensor tensor4 = torch::from_blob(ol4,   {1, 4, [ols1[3] intValue], [ols2[3] intValue]}, at::kFloat);
    at::Tensor tensor5 = torch::from_blob(ol5,   {1, 2, [ols1[4] intValue], [ols2[4] intValue]}, at::kFloat);
    at::Tensor tensor6 = torch::from_blob(ol6,   {1, 4, [ols1[5] intValue], [ols2[5] intValue]}, at::kFloat);
    at::Tensor tensor7 = torch::from_blob(ol7,   {1, 2, [ols1[6] intValue], [ols2[6] intValue]}, at::kFloat);
    at::Tensor tensor8 = torch::from_blob(ol8,   {1, 4, [ols1[7] intValue], [ols2[7] intValue]}, at::kFloat);
    at::Tensor tensor9 = torch::from_blob(ol9,   {1, 2, [ols1[8] intValue], [ols2[8] intValue]}, at::kFloat);
    at::Tensor tensor10 = torch::from_blob(ol10, {1, 4, [ols1[9] intValue], [ols2[9] intValue]}, at::kFloat);
    at::Tensor tensor11 = torch::from_blob(ol11, {1, 2, [ols1[10] intValue], [ols2[10] intValue]}, at::kFloat);
    at::Tensor tensor12 = torch::from_blob(ol12, {1, 4, [ols1[11] intValue], [ols2[11] intValue]}, at::kFloat);
      
    torch::autograd::AutoGradMode guard(false);
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    auto bboxlist = _impl.forward({tensor1, tensor2, tensor3, tensor4, tensor5, tensor6, tensor7, tensor8, tensor9, tensor10, tensor11, tensor12}).toTensor();
      
    float* output = bboxlist.data_ptr<float>();
    if (!output) {
      return nil;
    }
      
    auto tensor_size = bboxlist.sizes();

    long long total_size = 1;
      for(long long size : tensor_size){
        total_size = total_size*size;
          
      }
    NSMutableArray* results = [[NSMutableArray alloc] init];
    for (int i = 0; i < total_size; i++) {
      [results addObject:@(output[i])];
    }
    return [results copy];
  } catch (const std::exception& exception) {
    NSLog(@"%s", exception.what());
  }
  return nil;
}

@end
