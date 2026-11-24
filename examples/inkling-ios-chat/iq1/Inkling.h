#import <Foundation/Foundation.h>

@interface Inkling : NSObject
// Call on a single worker queue. The update block runs on that queue.
- (NSString *)loadFromPath:(NSString *)path;
- (NSString *)replyTo:(NSString *)message
                since:(NSTimeInterval)start
               update:(void (^)(NSString *text, NSString *metrics))update;
@end
