#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface SSFDPreProcessingModule : NSObject

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath
    NS_SWIFT_NAME(init(fileAtPath:))NS_DESIGNATED_INITIALIZER;
+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;
- (nullable NSArray<NSNumber*>*)preprocessImage_ssfd:(void*)imageBuffer
                                           im_height:(NSNumber*)im_height
                                           im_width: (NSNumber*)im_width
                                    NS_SWIFT_NAME(predict(image:image_height:image_width:));

@end

@interface SSFDPostProcessingModule : NSObject

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath
    NS_SWIFT_NAME(init(fileAtPath:))NS_DESIGNATED_INITIALIZER;
+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;
- (nullable NSArray<NSNumber*>*)postprocessImage_ssfd: (void*)ol1
                                                   ol2:(void*)ol2
                                                   ol3:(void*)ol3
                                                   ol4: (void*)ol4
                                                   ol5: (void*)ol5
                                                   ol6: (void*)ol6
                                                   ol7: (void*)ol7
                                                   ol8: (void*)ol8
                                                   ol9: (void*)ol9
                                                   ol10: (void*)ol10
                                                   ol11: (void*)ol11
                                                   ol12: (void*)ol12
                                                   ols1: (NSArray<NSNumber*>*)ols1
                                                   ols2: (NSArray<NSNumber*>*)ols2
                                                    NS_SWIFT_NAME(predict(ol1:ol2:ol3:ol4:ol5:ol6:ol7:ol8:ol9:ol10:ol11:ol12:ols1:ols2:));

@end

NS_ASSUME_NONNULL_END
