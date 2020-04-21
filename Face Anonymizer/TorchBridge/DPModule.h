//
//  DPModule.h
//  Face Anonymizer
//
//  Created by Jose Bouza on 4/10/20.
//  Copyright © 2020 Jose Bouza. All rights reserved.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface DeepPrivacyModule : NSObject

- (nullable instancetype)initWithFileAtPath:(NSString*)filePath
    NS_SWIFT_NAME(init(fileAtPath:))NS_DESIGNATED_INITIALIZER;
+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;
- (nullable NSArray<NSNumber*>*)deepPrivacyGenerator:(void*)imageBuffer
                                           im_width : (NSNumber*)im_width
                                           im_height: (NSNumber*)im_height
                                           keypoints:(void*)keypoints
                                           bbox: (void*)bbox
                                           n: (NSNumber*)n
                                    NS_SWIFT_NAME(predict(image:image_width:image_height:keypoints:bbox:n:));

@end

NS_ASSUME_NONNULL_END
