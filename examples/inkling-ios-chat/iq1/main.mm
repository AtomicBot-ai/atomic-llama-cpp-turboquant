#import <UIKit/UIKit.h>
#import <TargetConditionals.h>
#import "Inkling.h"

static UILabel *label(UIFontTextStyle style, UIColor *color) {
    UILabel *view = [UILabel new];
    view.font = [UIFont preferredFontForTextStyle:style];
    view.adjustsFontForContentSizeCategory = YES;
    view.textColor = color;
    view.numberOfLines = 0;
    return view;
}

@interface ChatController : UIViewController <UITextViewDelegate>
@end

@implementation ChatController {
    UITextView *_input, *_answer;
    UIButton *_send;
    UILabel *_status, *_metrics;
    Inkling *_engine;
    dispatch_queue_t _worker;
    BOOL _ready, _busy;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = UIColor.systemBackgroundColor;
    self.title = @"Inkling";
    _engine = [Inkling new];
    _worker = dispatch_queue_create("inkling.inference", DISPATCH_QUEUE_SERIAL);
    _status = label(UIFontTextStyleSubheadline, UIColor.secondaryLabelColor);
    _status.text = @"Loading Inkling IQ1_S…";
    _answer = [UITextView new];
    _answer.editable = NO;
    _answer.accessibilityLabel = @"Response";
    _answer.accessibilityIdentifier = @"answer";
    _answer.font = [UIFont preferredFontForTextStyle:UIFontTextStyleBody];
    _answer.adjustsFontForContentSizeCategory = YES;
    _answer.textContainerInset = UIEdgeInsetsZero;
    _answer.textContainer.lineFragmentPadding = 0;
    _metrics = label(UIFontTextStyleFootnote, UIColor.secondaryLabelColor);
    _metrics.accessibilityIdentifier = @"metrics";

    UILabel *inputLabel = label(UIFontTextStyleHeadline, UIColor.labelColor);
    inputLabel.text = @"Your question";
    _input = [UITextView new];
    _input.accessibilityLabel = inputLabel.text;
    _input.accessibilityIdentifier = @"prompt";
    _input.font = [UIFont preferredFontForTextStyle:UIFontTextStyleBody];
    _input.adjustsFontForContentSizeCategory = YES;
    _input.backgroundColor = UIColor.secondarySystemBackgroundColor;
    _input.layer.cornerRadius = 12;
    _input.textContainerInset = UIEdgeInsetsMake(10, 8, 10, 8);
    _input.delegate = self;
    _send = [UIButton buttonWithType:UIButtonTypeSystem];
    UIButtonConfiguration *config = UIButtonConfiguration.filledButtonConfiguration;
    config.title = @"Send";
    config.cornerStyle = UIButtonConfigurationCornerStyleMedium;
    _send.configuration = config;
    _send.accessibilityIdentifier = @"send";
    [_send addTarget:self action:@selector(sendQuestion) forControlEvents:UIControlEventTouchUpInside];
    [self refreshButton];

    UIStackView *stack = [[UIStackView alloc] initWithArrangedSubviews:@[_status, _answer, _metrics, inputLabel, _input, _send]];
    stack.axis = UILayoutConstraintAxisVertical;
    stack.spacing = 10;
    stack.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:stack];
    UILayoutGuide *safe = self.view.safeAreaLayoutGuide;
    NSLayoutConstraint *inputHeight = [_input.heightAnchor constraintEqualToConstant:76];
    inputHeight.priority = UILayoutPriorityDefaultHigh;
    [NSLayoutConstraint activateConstraints:@[
        [stack.topAnchor constraintEqualToAnchor:safe.topAnchor constant:12],
        [stack.leadingAnchor constraintEqualToAnchor:safe.leadingAnchor constant:20],
        [stack.trailingAnchor constraintEqualToAnchor:safe.trailingAnchor constant:-20],
        [stack.bottomAnchor constraintEqualToAnchor:self.view.keyboardLayoutGuide.topAnchor constant:-12],
        inputHeight, [_send.heightAnchor constraintGreaterThanOrEqualToConstant:44]
    ]];
    [_answer setContentHuggingPriority:UILayoutPriorityDefaultLow - 1 forAxis:UILayoutConstraintAxisVertical];
    dispatch_async(_worker, ^{
        NSString *dir = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
        NSString *error = [self->_engine loadFromPath:[dir stringByAppendingPathComponent:@"Inkling-2-Mini-AD-IQ1_S.gguf"]];
        dispatch_async(dispatch_get_main_queue(), ^{
            self->_ready = !error;
            self->_status.text = error ?: @"Inkling IQ1_S · Ready";
            [self refreshButton];
        });
    });
}

- (NSString *)question {
    return [_input.text stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
}

- (void)refreshButton {
    _send.enabled = _ready && !_busy && self.question.length > 0;
}

- (void)textViewDidChange:(UITextView *)textView {
    [self refreshButton];
}

- (void)sendQuestion {
    if (!_send.enabled) return;
    NSTimeInterval start = NSProcessInfo.processInfo.systemUptime;
    NSString *question = self.question;
    _busy = YES;
    [self refreshButton];
    [_input resignFirstResponder];
    _answer.text = @"";
    _metrics.text = @"";
    _status.text = @"Generating…";
    dispatch_async(_worker, ^{
        NSString *status = [self->_engine replyTo:question since:start update:^(NSString *text, NSString *metrics) {
            dispatch_async(dispatch_get_main_queue(), ^{
                self->_answer.text = text;
                if (text.length) [self->_answer scrollRangeToVisible:NSMakeRange(text.length - 1, 1)];
                self->_metrics.text = metrics;
            });
        }];
        dispatch_async(dispatch_get_main_queue(), ^{
            self->_status.text = status ?: @"Response complete";
            self->_busy = NO;
            [self refreshButton];
        });
    });
}
@end

@interface ChatScene : UIResponder <UIWindowSceneDelegate>
@property(nonatomic, strong) UIWindow *window;
@end
@implementation ChatScene
- (void)scene:(UIScene *)scene willConnectToSession:(UISceneSession *)session options:(UISceneConnectionOptions *)options {
    self.window = [[UIWindow alloc] initWithWindowScene:(UIWindowScene *)scene];
    self.window.rootViewController = [[UINavigationController alloc] initWithRootViewController:[ChatController new]];
    [self.window makeKeyAndVisible];
}
@end

@interface ChatApp : UIResponder <UIApplicationDelegate>
@end
@implementation ChatApp
@end

int main(int argc, char **argv) {
#if TARGET_OS_SIMULATOR
    setenv("GGML_METAL_SIM_SYNC_READBACK", "1", 0);
    setenv("LLAMA_EXPERIMENTAL_INKLING_SIM_CPU", "parallel", 0);
    setenv("GGML_METAL_SHARED_BUFFERS_ENABLE", "1", 0);
    setenv("GGML_METAL_NO_RESIDENCY", "1", 0);
    setenv("GGML_METAL_FUSION_DISABLE", "1", 0);
    setenv("GGML_METAL_CONCURRENCY_DISABLE", "1", 0);
#endif
    @autoreleasepool {
        return UIApplicationMain(argc, argv, nil, NSStringFromClass([ChatApp class]));
    }
}
